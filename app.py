"""Nalla Chatbot - Streamlit + Gemini (streaming, personas, chat download)."""
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

st.set_page_config(page_title="Nalla Chatbot", page_icon="💬", layout="centered")

# ---------------------------------------------------------------- config
PERSONAS = {
    "😊 Friendly Buddy": (
        "You are a warm, friendly assistant. Reply in the same language the user writes in "
        "(Tamil, Tanglish or English). Keep answers clear and short unless asked for detail."
    ),
    "🎓 Study Helper": (
        "You are a patient study helper for engineering students. Explain concepts step by step "
        "with simple examples, formulas where useful, and finish with a one-line summary. "
        "Reply in the user's language (Tamil, Tanglish or English)."
    ),
    "💻 Coding Mentor": (
        "You are a coding mentor. Give working, minimal code in fenced blocks, explain what it does "
        "in a few lines, and point out common mistakes. Reply in the user's language."
    ),
    "✉️ Writing Assistant": (
        "You help write and improve emails, resumes, reports and messages. Ask at most one clarifying "
        "question, otherwise just draft. Keep the tone professional unless told otherwise."
    ),
}
SUGGESTIONS = [
    "Explain Fourier transform simply",
    "Write a leave letter to my HOD",
    "Python la list vs tuple difference enna?",
    "Give me interview tips for a fresher",
]

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 820px;}
.hero {text-align:center; padding: .5rem 0 1rem 0;}
.hero h1 {font-size: 2.1rem; margin-bottom: .2rem;}
.hero p {color: #8b93a1; margin: 0;}
[data-testid="stChatMessage"] {border-radius: 14px; padding: .8rem 1rem;}
.stButton>button {border-radius: 999px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def get_api_key() -> str | None:
    try:  # Streamlit Cloud secrets
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


@st.cache_resource
def get_client(key: str):
    return genai.Client(api_key=key)


def to_contents(messages):
    return [
        types.Content(role="user" if m["role"] == "user" else "model",
                      parts=[types.Part(text=m["content"])])
        for m in messages
    ]


def stream_reply(client, model, system, temperature, messages):
    stream = client.models.generate_content_stream(
        model=model,
        contents=to_contents(messages),
        config=types.GenerateContentConfig(system_instruction=system, temperature=temperature),
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def transcript(messages) -> str:
    lines = [f"Nalla Chatbot transcript - {datetime.now():%Y-%m-%d %H:%M}", ""]
    for m in messages:
        who = "You" if m["role"] == "user" else "Bot"
        lines += [f"{who}: {m['content']}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    persona = st.selectbox("Persona", list(PERSONAS))
    model = st.text_input("Model", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    temperature = st.slider("Creativity", 0.0, 1.5, 0.7, 0.1)
    st.divider()
    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.session_state.messages:
        st.download_button("⬇️ Download chat", transcript(st.session_state.messages),
                           file_name="chat.txt", use_container_width=True)
    st.caption(f"{len(st.session_state.messages)} messages")

# ---------------------------------------------------------------- main
st.markdown('<div class="hero"><h1>💬 Nalla Chatbot</h1>'
            f'<p>{persona} · powered by Gemini</p></div>', unsafe_allow_html=True)

api_key = get_api_key()
if not api_key:
    st.error("GEMINI_API_KEY illa. `.env` la (or Streamlit Secrets la) key add pannunga. "
             "Key: aistudio.google.com/apikey")
    st.stop()

if not st.session_state.messages:
    st.write("Enna pathi pesalam? Edhavadhu onna click pannunga 👇")
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, key=f"sg{i}", use_container_width=True):
            st.session_state.pending = s
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
        st.markdown(m["content"])

user_input = st.chat_input("Message type pannunga...") or st.session_state.pending
st.session_state.pending = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    with st.chat_message("assistant", avatar="🤖"):
        try:
            reply = st.write_stream(stream_reply(
                get_client(api_key), model, PERSONAS[persona], temperature,
                st.session_state.messages))
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.session_state.messages.pop()  # remove unanswered user msg
            st.error(f"Error: {e}")

import tensorflow_hub as hub
embed = hub.Module("https://tfhub.dev/google/"

"universal-sentence-encoder/2")
embedding = embed(["The quick brown fox jumps over the lazy dog"])
embedding
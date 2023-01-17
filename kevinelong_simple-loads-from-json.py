import json

text = """[
    "larry", 
    "moe", 
    "curly"
]"""

print(len(text))

data = json.loads(text)

print(len(data))
# print(data)
for name in data:
    print(f"NAME={name}")

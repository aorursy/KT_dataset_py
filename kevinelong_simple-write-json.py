import json

data = {
    "k1": [
        "v1a",
        "v1b",
        "v1c"
    ],
    "k2": [
        "v1a",
        "v1b",
        "v1c"
    ]
}

print(data)
print(len(data))

text = json.dumps(data)

print(len(text))
print(text)

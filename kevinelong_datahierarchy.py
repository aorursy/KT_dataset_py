items = []

lookup = {}

data = [
    {
        "first_name" : "kevin",
        "last_name" : "long",
        "favorite_game": "one pocket billiards",
        "pets": [
            "toby the lab dane mutt",
            "bianca the totamese",
            "fineley the bengal",
        ]
    },
    {
        "first_name" : "Ambrosia",
        "last_name" : "Griffin",
        "favorite_game": "Monopoly",
        "pets": [
            "tobi the yorkie"
        ]     
    }
]
item1 = data[1]
last_name = item1["last_name"]
print(last_name)
print(data[0]["favorite_game"])
print(data[0]["pets"][-1])
print(data[1]["pets"][0])

for person in data:
    for pet in person["pets"]:
        print(f"{pet} belongs to {person['first_name']}.")
rows = [
    ['.', '.', '.'],
    ['.', '.', '.'],
    ['.', '.', '.']
]

print(rows)

rows[1][1] = "X"

print(rows)

# GOAL
# . . .
# . X .
# . . .

# for row in rows:
#     for letter in row:
#         print(letter, end=" ")
#     print("")
topics = {
    "colors" : ["red","green","blue"],
    "sizes" : ["XS", "Small", "Medium", "Large", "Xtra Large"],
    "gender": ["unisex", "Male", "Female"]
}
# print all values in all topics

# treate a dictionary as a list you get the keys
# but then you use the keys to get the values

# for key in topics:
#     value = topics[key]
#     print(key)
#     for item in value:
#         print("\t", item)

data = [
    "cherry",
    "apple",
    "pear",
    "apple",
    "cherry",
    "apple",
    "apple",
    "pear",
    "cherry",
]

summary = {
    "apple" : 0,
    "cherry" : 0,
    "pear" : 0,
}

#L1 TODO
# use a for loop
# inside we will increase the count, accessed with square brackets and the key += 1


print(summary)
#L2 XC KNOWLEDGE - start the above with an empty summary.
colors = ["red", "green"]
if "blue" not in colors:
    colors.append("blue")
    
color_count = {"red":1, "green":1}
if "blue" not in color_count:
    color_count["blue"] = 1
    
print(color_count)
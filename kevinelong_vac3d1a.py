# By Kevin Ernest Long
x = 6
y = 7
print(x * y)

a = 2
b = 4
print(b / a)

half = 0.5
third = 1/3
print(half)
print(third)
quantity = 123.333333333
print(f"The Quantity is { quantity }.")


isCool = True
tooCool = False

isGreater = 5 > 5
isEqual = 5 == 5
print(isGreater)
print(isEqual)

print( "ABC" == "ABC")

print("ZZZ")
a:int = 123
a = 888
a = 111
print(a)
x = 3 + 7 * 9 / 4
print(x)
print (12 * 12)
numbers = [12,34,56]

names = [
    "Bob",
    "Carol",
    "Ted",
    "Alice",
    "Kevin"
]

names.append("Nina")
names.append("John")
names.append("Bianca")

names.remove("Carol")

print(numbers)
print(names)

print(names[0])
print(names[1])

print(names[3])

print(names[-1])

print(names[-2])
print(names[2])

phrase_book = {
    "alpha" : "A as in Alpha.",
    "bravo" : "B as in Bravo.",
    "charlie" : "C as in Charlie."
}

phrase = phrase_book["alpha"]

del phrase_book["bravo"]

print(phrase)

phrase_book["david"] = "D as in David."

print(phrase_book)

print(phrase_book["david"])

word_dict = { "WORD" : "DEFINITION"}
print(word_dict["WORD"])
print(97 == 65)
print(ord("A")) #ORDINAL VALUE i.e. the code for the letter
print(ord("a")) #ORDINAL VALUE i.e. the code for the letter

print("A" == "a")
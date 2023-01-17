for i in range(32,127):
    print( str(i) + " = DECIMAL:" + chr( i ) + ' BINARY:' + bin(i) + ' HEX:' + hex(i) )

#ASCII - Plain Ordinary English Text
#American Standard Code for Information Interchange

# ascii = {
#     "\r": 10,
#     "\n": 13,
#     " ": 32,
#     "A": 65,
#     "B": 66,
#     ...
# }

# {
#     1: "00000001",
#     2: "00000010",
#     3: "00000011",
#     ...
# }
# Variable(label, identifier, atribute name, property, fieled, column, cell): A named reference to a value
x = "ABC"
y = "ABC"
y = "ABCX"
Z26 = 987
y
# Simple Types

# Number: numbers come in two flavors
# Integer: whole numbers (e.g. 1,2,3)
i = 12
# Float: A floating-point/decimal number can hold fractional values (e.g. 1.0, 0.25 13.79)
f = 33.33
half = 1.0 / 2.0

# String: An ordered sequence of characters (letter)
student_name = "Alice Jones"
# List/Array/Vector: A list is an ordered collection of values that can be referenced by position in the list [0,1,2,3,...]
# 0 is the index of the first item, 1 is the second item 2 is the third item etc
# -1 is the last item -2 is the second to the last item
people = ["Bob", "Carol", "Ted", "Alice"]
first_person = people[0]
last_person = people[-1]

quarterly_sales_for_year = [100, 75, 50, 200]
q3 = quarterly_sales_for_year[2]

#Dictionary/Object/HashTable: a dictionary is an unordered collection of values that can be accessed by a name known as a key like a phone book or library
phone_book = {
    "Bob": "555-555-2222", 
    "Carol": "555-555-3333", 
    "Ted": "555-555-4444", 
    "Alice": "555-555-1111"
}
alice_number = phone_book["Alice"]

is_a_student = True
is_cheating = False
# Math Expressions and Operators
sum = 2 + 2
product = 3 * 4
fraction = 1.0 / 2.0
modulus_remainder = 5 % 4

#Boolean Expressions
is_greater = 7 > 9
is_equal = (9 == (4 + 5))

# CONDITIONALS
# if, elif, else
state = "bad"
if state == "good":
    destination = "heaven"
elif state == "bad":
    destination = "hell"
else:
    destination = "purgatory"
# Example

alive = True
lives = 9

while alive:
    
    if alive:
        print("Dance " + str(lives))

    lives = lives - 1
    
    if lives == 0:
        alive = False

people = ["Bob", "Carol", "Ted", "Alice"]
for person in people:
    print(person)

people
#built in functions
print(len("word, again"))
print(len(str(3 * 4)))
## Functions are a sequence of code instructions you can call by name to avoid copy paste modify.
# like a useful formula

def celcius(farenheit):
    output = ((farenheit - 32) * 5) / 9
    return output

# D on't
# R epeat
# Y ourself

result = celcius(100)
print ("100 -> " + str(result))


class Person(object):
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name
    def full_name(self):
        return self.first_name + " " + self.last_name



#Encapsulation

class Group(object):
    def __init__(self, people):
        self.people = people
    def show(self):
        for person in self.people:
            print(person.full_name())

            
p = Person("Kevin", "Long")
print(p.first_name)
print(p.full_name())

# Note use of p from above
g = Group([
    p,
    Person("Ashley", "Ford"),
    Person("Bob", "Dobbs")
])

k = g.people[0]
print(k.first_name)

g.show()



data = [12,34,56,67,89]
first_two = data[:2]
first_two
print(first_two)

last_two = data[-2:]
print(last_two)

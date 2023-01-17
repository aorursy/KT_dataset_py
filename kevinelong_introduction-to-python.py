# Variable: A named reference to a value
quantity = 144
# Number: numbers come in two flavors
# Integer: whole numbers (e.g. 1,2,3)
i = 12
# Float: A floating-point/decimal number can hold fractional values (e.g. 1.0, 0.25 13.79)
f = 33.33
half = 1.0 / 2.0
# String: An ordered sequence of characters (individual letters, numbers, or symbols)

student_name = "Alice Jones"
# List: A list is an ordered collection of values that can be referenced by position in the list [0,1,2,3,...]
# 0 is the index of the first item, 1 is the second item 2 is the third item etc
# -1 is the last item -2 is the second to the last item
people = ["Bob", "Carol", "Ted", "Alice"]
first_person = people[0]
last_person = people[-1]

quarterly_sales_for_year = [100, 75, 50, 200]
q3 = quarterly_sales_for_year[2]
#Dictionary: a dictionary is an unordered collection of values that can be accessed by a name known as a key like a phone book or library
phone_book = {
    "Bob": "555-555-2222", 
    "Carol": "555-555-3333", 
    "Ted": "555-555-4444", 
    "Alice": "555-555-1111"
}
alice_number = phone_book["Alice"]
#Boolean: A Boolean is a binary logical value. e.g. True or False
is_a_student = True
is_cheating = False
###
# EXPRESSIONS AND OPERATORS
###
sum = 2 + 2
product = 3 * 4
fraction = 1 / 2
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
# While

happy = True
claps = 0
limit = 2

while happy:
    if claps <= limit:
        print(f"Clap {claps}")
    else:
        happy = False
        
    claps = claps + 1  #  CLEAREST
    # claps += 1       #  SUCCINCT & AVAILABLE IN PYTHON
    # claps++          #  NOT AVAILABLE IN PYTHON

# For
people = ["Bob", "Carol", "Ted", "Alice"]
for person in people:
    print(person)
###
# DEFINING FUNCTIONS
###

#built in functions
print(len("word, again"))
print("BINARY:",  bin(3 * 4))
print(len(str(bin(3 * 4))))
## Functions are a sequence of code instructions you can call by name to avoid copy paste modify.
# like a useful formula

def celcius(farenheit):
    output = ((farenheit - 32) * 5) / 9
    return output

# D on't
# R epeat
# Y ourself

print ("100 -> " + str(celcius(100)))
print (f"100 -> {str(celcius(100))} .")
oneThroughFive = range(1,6)
# up to but not including the number on the right.
# [1,2,3,4,5]

zeroToFour = range(0,5) 
# Up to but not including the number on the right.
#  [0,1,2,3,4]
print(list(oneThroughFive))
print(list(zeroToFour))

###
# Classes and Modules
###
class Person(object):
    #CONSTRUCTOR
    def __init__(self, first_name, last_name):
        #ATTRIBUTES - PROPERTIES
        self.first_name = first_name
        self.last_name = last_name
    #METHOD
    def full_name(self):
        return self.first_name + " " + self.last_name

p = Person("Kevin", "Long")
print( p.full_name() )
#Encapsulation - Cell Like

# class Employee(person):
#     salary = 999
    
class Group(list):
    def __init__(self, people):
        self.people = people

    def show(self):
        for person in self.people:
            print(person.full_name())


p = Person("Kevin", "Long")
print(p.first_name)

g = Group([
    p,
    Person("Ashley", "Ford"),
    Person("Bob", "Dobbs"),
])

k = g.people[0]
print(k.first_name)

g.show()
#total
def total(data):
    total = 0
    for value in data:
        total += value
    return total

print(total([123,456,789]))
# minimum
# maximum
# average

#total
def average(data):
    total = 0
    count = 0
    for value in data:
        total += value
        count += 1
    return total / count

print(average([1,1,2]))
def minimum(data):
    counter = 0
    for value in data:
        if counter == 0:
            lowest = value
        elif value < lowest:
            lowest = value
        counter += 1
        
    return lowest
print(minimum([441,31,52]))

def maximum(data):
    counter = 0
    biggest = 0
    for value in data:
        if counter == 0:
            biggest = value
        elif value > biggest:
            biggest = value
        counter += 1
        
    return biggest

print(maximum([441,1131,952]))

def maximum(data):
    if len(data) > 0:
        biggest = data[0]
    else:
        biggest = 0
        
    for value in data:
        if value > biggest:
            biggest = value
    return biggest

print(maximum([441,1131,952]))

# ##
# VARIABLES AND DATA TYPES
###
# Variable: A named reference to a value
quantity = 144
# Data Type: A Specific kind of variable (e.g. Integer, String, List)
# Number: numbers come in two flavors
# Integer: whole numbers (e.g. 1,2,3)
i = 12
# Float: A floating-point/decimal number can hold fractional values (e.g. 1.0, 0.25 13.79)
f = 33.33
half = 1.0 / 2.0
# String: An ordered sequence of characters
student_name = "Alice Jones"
# List: A list is an ordered collection of values that can be referenced by position in the list [0,1,2,3,...]
# 0 is the index of the first item, 1 is the second item 2 is the third item etc
# -1 is the last item -2 is the second to the last item
people = ["Bob", "Carol", "Ted", "Alice"]
first_person = people[0]
last_person = people[-1]

quarterly_sales_for_year = [100, 75, 50, 200]
q3 = quarterly_sales_for_year[2]
# Dictionary: a dictionary is an unordered collection of values that can be accessed by a name known as a key like a phone book or library
phone_book = {
    "Bob": "555-555-2222",
    "Carol": "555-555-3333",
    "Ted": "555-555-4444",
    "Alice": "555-555-1111"
}
alice_number = phone_book["Alice"]

# Boolean: A Boolean is a binary logical value. e.g. True or False
is_a_student = True
is_cheating = False
###
# EXPRESSIONS AND OPERATORS
###

sum = 2 + 2
product = 3 * 4
fraction = 1 / 2
modulus_remainder = 5 % 4

# Boolean Expressions
is_greater = 7 > 9
is_equal = (9 == (4 + 5))
###
# CONTROL FLOW
###
# CONDITIONALS
# if, elif, else

state = "bad"

if state == "good":
    destination = "heaven"
elif state == "bad":
    destination = "hell"
else:
    destination = "purgatory"
# WHILE LOOPS

alive = True
lives = 9

while alive:
    if alive:
        print("Dance " + str(lives))
    else:
        alive = False
    lives = lives - 1  # CLEAREST
    # lives -= 1 AVAILABLE
    # lives-- #NOT AVAILABLE

    if lives == 0:
        alive = False
#FOR LOOPS

people = ["Bob", "Carol", "Ted", "Alice"]
for person in people:
    print(person)

for i in range(13):
    print(f"i is {i}")
    
###
# CODE ORGANIZATION FOR CLARITY AND REUSE
# ---
# DEFINING FUNCTIONS
###

# D on't
# R epeat
# Y ourself
# built in functions
print(len("word, again"))
print("BINARY:", bin(3 * 4))
print(len(str(bin(3 * 4))))
## Functions are a sequence of code instructions you can call by name to avoid copy paste modify.
# like a useful formula

def celcius(farenheit):
    output = ((farenheit - 32) * 5) / 9
    return output

print("100 -> " + str(celcius(100)))
print(f"100 -> {str(celcius(100))} .")
# MODEL THE REAL WORLD WITH CLASSES

class Animal: #NOUN (PEOPLE/PLACE/THING )
    
    def __init__(self, kind, noise, name = ""):
        # ATTRIBUTE, ADJECTIVE PROPERTY
        self.kind = kind
        self.noise = noise
        self.name = name
        
    def speak(self): #VERB-ACTION WORDS
        if self.name == "":
            print(f"{self.kind} goes {self.noise}!")
        else:
            print(f"{self.name} goes {self.noise}!")
            
        
duck = Animal("Duck", "quack")
duck.speak()


class Cat(Animal): # Cat is an Animal inheritance, creating sub-class out super-class/parent is Animal
    def __init__(self, name):
        super().__init__("Cat", "meow", name)
        
duck = Animal("Duck", "quack")
duck.speak()

bianca = Cat("Bianca")
bianca.speak()

f = Cat("Finnley")
f.speak()

#DUCK TYPING (if it looks like and quaks like a duck its a duck)
pets = [duck, bianca, f]
for creature in pets:
    creature.speak()
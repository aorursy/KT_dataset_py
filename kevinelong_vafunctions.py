def addTwo(a:int = 0, b:int = 0) -> int:
        return a + b
    
addResult1 = addTwo()

def addTwo(a, b):
    return a + b

addResult2 = addTwo(2, 4)

print(addResult1)
print(addResult2)

print(addTwo(8, 9))

x = 8
y = 9
z = addTwo(x,y)
print(z)
def twice(text: str) -> str:
    return f"{text}---{text}"

twiceResult = twice("abc")
print(twiceResult)

# for
# if

data = [456,234,987,123,678]

def get_lowest(data):
    lowest = data[0] # FIRST ONE
    # ... YOUR CODE HERE
    for item in data:
        if lowest > item:
            lowest = item
    return lowest

result = get_lowest(data)
print(result)
data = [456,234,987,123,678]

def get_average(data):
    total = 0
    quantity = len(data)
    
    #... YOUR CODE GOES HERE
    for item in data:
        total = total + item
#         total += item #short version of the line above
        
#     total = sum(data)
    
    average = total / quantity
    
#     return sum(data) / len(data)
    
    return average


result = get_average(data)
print(result)
data = [456,234,987,123,678]

def get_average(data):
    total = 0
    quantity = len(data)
    #... YOUR CODE GOES HERE
    for item in data:
        total = total + item
    average = total / quantity
    
    return average
result = get_average(data)
print(result)
# looping through a dict loops through the keys
# using the key you can get the value
people = {
    "KEL" : "Kevin Ernest Long",
    "KJL" : "Kay Jowan Long"
}
for key in people:
    value = people[key]
    print(key, value)
def count_change(denomination_quantities):
    total = 0
    # TODO put code here

    return total


denomination_quantities = {
    1: 3,
    5: 1,
    10: 1,
    25: 3,
}

print(count_change(denomination_quantities))
# Expected result: 93

def count_change(denomination_quantities):
    total = 0
    # TODO put code here

    for denomination in denomination_quantities:
        print(f"D={denomination}")

        quantity = denomination_quantities[ denomination ]
        print(f"Q={quantity}")
        
        subtotal = quantity * denomination
        total += subtotal
        print(quantity, denomination, subtotal, total)
        
    return total


denomination_quantities = {
    1: 3,
    5: 1,
    10: 1,
    25: 3,
}

print(count_change(denomination_quantities))
# Expected result: 93

print("What is your name?")
user_data = input()
print(f"Hello {user_data}!")

text = "0"
total = 0

while text != "":
    print("Enter an integer to add to the total. Blank line to quit.")
    text = input()
    if text != "":
        total  += int(text)
    print(f"Total: {total}")
    
print("Bye!")
    
#self == instance == real == in memory

# Nouns -people, place, thing
class Animal:
    def __init__(self, noise, legs = 4):
        #adjectives - property - attribute
        self.noise = noise
        self.legs = legs
        #self is like me/myself/I
        
    #Verbs - methods - function inside of a class e.g. run, print
    def speak(self):
        print(self.noise)
    def walk(self):
        for _ in range(self.legs):
            print("thump")
            
a = Animal("Grunt!")
a.speak()
a.walk()

class Dog(Animal): # IS A..
    pass
#     def __init__(self):
#         self.noise = "Woof!"
#         self.legs = 4
        
d = Dog("Bark!", 3)
d.speak()
d.walk()

name = None
if name is None:
    print("Who are you?!?!?")
class Player:
    def __init__(self, name):
        self.name = name
        self.score = 0
    
class Game:
    def __init__(self):
        self.player_list = [ 
            Player("A"),
            Player("B")
        ]
        #COMPOSITION a game is made up a a collection of players - HAS A
        
    def show_stats(self):
        for p in self.player_list:
            print(f"Player {p.name} has {p.score} points!")
            
g = Game()
g.show_stats()

name = "ABC"
for letter in name:
    n = ord(letter)
    print(letter, n, bin(n))

for d in [0b1001011,0b1000101,0b1010110,0b1001001,0b1001110]:
    print(chr(d), end="")
t = type("AAA")
print(t)
print( isinstance("ZZZ", t))
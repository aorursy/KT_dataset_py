import random
def roll(quantity = 3, sides = 6):
    total = 0
    for index in range(1 , quantity + 1):
        total += random.randint(1, sides)
    # random.randint (min value, max possible value)  between 3 and 18
    return total

#     loop through a range
print(roll())
print (roll(4,2)) #1 for heads, 2 for tails  4 or 8

def roll_many(time = 100):
    data = []
    for index in range (0, time):
        data.append(roll())
    return data

print(roll_many(10))
#example of data = { 3:1, 4:5, 11:23, 12:50, 18:1}
#keys will be range 3-18


def generate (epochs = 100, quantity = 3, sides = 6 ):
    data = {}
    for index in range (0, epochs):
        val = roll(quantity, sides)
        #when checking a single value, is only checking the keys
        if (val in data):
            data[val] += 1
        else:
          data[val] = 1  
    return data

data = generate(10)
for k in data:
    msg =  "{0} was rolled {1} times".format(k, data[k])
    print (msg)

    
#scale it to fit the screen
#takes a parameter called data that is a dict/hash prints and ascii chart with hashes
def chart (data):
    #  3|#### (4)
    #get maximum key
    biggest = 0
    for key in data:
        if biggest < key:
            biggest = key
    #get maximum val
    biggestVal = 0
    for key in data:
        if biggestVal < data[key]:
            biggestVal = data[key]
            
    mult = 100/biggestVal

    for i in range (0, biggest):
        if i in data:
            n = int(data[i] * mult)
            
            row = "{0}{1}|{2} ({3})".format((i<10)*" ", i, n *"#",data[i])
        else:
            x = (i<10)*" "
            row = f"{x}{i}| (0)"
        print (row)

quantity = 4
sides = 6
    
chart(generate(50, quantity, sides))           
        
    
import random
#computer will pick a number between 1 and 10
#snake case in python variables snake_case
solution = random.randint (1, 10+1)
game_over = False
while game_over == False:
    print ("Guess a number:")
    user_input = input()
    value = int(user_input)
    if value < solution:
        print ("Guess again higher")
    if value > solution:
        print ("Guess again lower")
    if value == solution:
        game_over = True
        print ("Congratulations!")
import random
#computer will pick a number between 1 and 10
#snake case in python variables snake_case
print ("Give a number:")
user_input = input()
value = int(user_input)
game_over = False
low = 1
high = 10+1
while game_over == False:

    guess = int(round ((low + high)/2,0))
    print (guess)
    if guess < value:
        print ("Computer, guess again higher")
        low = guess + 1
    if guess > value:
        print ("Computer, guess again lower")
        high = guess - 1
    if value == guess:
        game_over = True
        print ("Congratulations!")
def total(data):
    total = 0
    for n in data:
        total = total + n   
    return total

def avg(data):
    total1 = total(data)
    n = len(data)
    return (total1 / n)

def minimum(data):
    n = len(data)
    if n == 0:
        return None
    
    temp = data[0]
    for i in range(n):
        
        if temp > data[i] :
            temp = data [i]
            
    return temp

def maximum(data):
    n = len(data)
    if n == 0:
        return None
    
    temp = data[0]
    for i in range(n):
        
        if temp < data[i] :
            temp = data [i]
            
    return temp

# minimum maximum average
# control slash is toggle comment octithorp 
print (total([123, 456, 789]))
print (minimum([123, 456, 789]))
print (avg([123, 456, 789]))
print (maximum([123, 456, 789]))
msg1 = f"100 -> {str(bin(100))} ."
msg2 = "100 -> {} .".format(str(bin(100)))
print(msg1)
print(msg2)

###
# Classes and Modules
###
class Person(object):
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name
    def full_name(self):
        return self.first_name + " " + self.last_name


#Encapsulation

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
    Person("Bob", "Dobbs")
])

k = g.people[0]
print(k.first_name)

g.show()
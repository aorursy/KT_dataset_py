# IDEAS WE WILL USE
name = "KEVIN" + " " + "LONG" #CONCATENATION
count = len(name) - 1

while count >= 0:
    letter = name[count ]
    print(letter)
    count = count - 1
    
#FUNCTION WE WANT TO WRITE
def reverse_me(text):
    reversed = "" # use a list instead in L2
    # TODO YOUR CODE HERE
    count = len(text) - 1
    while count >= 0:
        letter = text[ count ]
        count = count - 1
        reversed = reversed + letter # L2 append to list
    return reversed # return with list join on "" empty string

print(reverse_me("ABC"))
#EXPECTED OUTPUT "CBA"

#L1 use string + conctenation
#L2 list/array and join on "" would be faster
def reverse_me(text):
    reversed = [] # use a list instead in L2
    
    count = len(text) - 1
    while count >= 0:
        letter = text[ count ]
        count = count - 1
        reversed.append(letter) 
    return "".join(reversed) #conver list back to  a string

print(reverse_me("ABC"))
#EXPECTED OUTPUT "CBA"
import random
#what we need to know
print(random.randint(1,6)) # generate random number betweeen 1 and 6

#PROBLEM
#roll multipl dice in a loop
def roll(quantity: int = 1, sides: int = 6) -> int:
    total = 0
    # TODO Put your code here.
    return total

# Test
print( roll(quantity=3,sides=6) )

#L2 who can we roll multiple sets and store the results in dict e,g, keys=3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
import random

def roll(quantity: int = 3, sides: int = 6) -> int:
    total = 0
    
    for n in range(quantity):
       total += random.randint(1,sides) 
    
    return total

print( roll(quantity=3,sides=6) )


#L2 who can we roll multiple sets and store the results in dict e,g, keys=3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
def make_data(limit = 100, quantity: int = 3, sides: int = 6):
    history = {} #how many time did each number occur?

    for n in range(limit):
        result = roll(quantity, sides)
        if result not in history:
            history[result] = 1
        else:
            history[result] += 1
    return history
history = make_data(500,3,6)
print(history)
def show_data(data, quantity: int = 3, sides: int = 6):
    
    for n in range(quantity,(quantity * sides)+1 ):
        if n in history:
            print(n, history[n] * "#")
        else:
            print(n)
show_data(history,3,6)
file_name = "products.csv"

output_file = open(file_name, "a")

line_list = [
    # ["ID", "NAME", "PRICE", "DESCRIPTION", "PHOTO_URL"],
    ["1003", "Meat Lovers", "39.99", "All the meats!!!", "http://www.example.com/photos/img_1001.png"],
    ["1004", "Veggie Delight", "39.99", "All the veg!!!", "http://www.example.com/photos/img_1002.png"],
]

for line in line_list:
    text = ",".join(line)
    output_file.write(text + "\n")
#  This is a comment that will be ignored.
6 + 7
"""
This is
a multiline
comment!
"""

print("Hello O'Reilly")

# this is also ignored

print('Goodbye "cruel" World')

a = 2
b = 3
x = a + b
y = a * b

print( x )
print( y )

print( 6 * 7 )
print(2 * (3 - 4) / (5 + 8))
# PMDAS - Order 

# * Mult
# / Division
# + Add
# - Sub

# TOP
PI = 3.14
SALES_TAX = 0.08

#AS NEEDED
q = 123
name = "Widget"
price = 12.34
print( f"QUANTIY: {q} NAME:{name} PRICE:{price} EXT: { q * price }")

isGreater = 3 < 2
print(isGreater)
isSame = 2 + 2 == 5
print(isSame)

print( 1 is not 1)
print( 1 is not 2)

names = ["Bob","Carol","Ted","Alice"]
print(names)
print(names[0])
print(names[3])
print(names[-1])
print(names[-2])
print(names[2])

print("AAA", "BBB")
print("CCC" + "DDD")
print("CCC" + " " + "DDD")

dudes = names[0] + names[2]
print(dudes)
print(names[0], names[2])
# Declare an Array of Strings
names: [str] = [] 
print(names)
# Append two items
names.append( "Bob" )
names.append( "Carol" )

print(names)
assert(len(names) == 2 ) # Verify

# Print Them Out
print( names[ 0 ] )
print( names[ 1 ] )

# Remove One Item
names.remove( "Bob" )
print(names)
assert( len(names) == 1 ) # Verify

names.append("Ted")
print(names)

numbers = [ 11,22,33]
print(numbers)
phrase_book: {str:str} = {
    "alpha" : "A as in Alpha",
    "bravo" : "B as in Bravo",
    "charlie" : "C is for Charlie",
}

phrase = phrase_book[ "bravo" ]

assert("B as in Bravo" == phrase)
print( phrase )
print(phrase_book)
coin_quantities:{int:int} = {}

coin_quantities[ 1 ] = 5 # We have 5 pennies
coin_quantities[ 5 ] = 4 # We have 4 nickels
coin_quantities[ 10 ] = 3 # We have 3 dimes
coin_quantities[ 25 ] = 2 # We have 2 quarters

quarter_count = coin_quantities[ 25 ]

assert( 2 == quarter_count )
print(coin_quantities)

# How to force display of exactly two decimal places:
# after the variable in the expression add:
# a colon, a decimal point, followed by the number desired and the letter f
# e.g.

price = 12.00
print(f'{price:.2f}')
######F=FORMAT###F=FLOATING_POINT_DECIMAL

price_fractional = 12.3333
print(f'{price_fractional:.2f}')

oneThroughFive = range(1,6)
# up to but not including the number on the right.
# [1,2,3,4,5]

zeroToFour = range(0,5) 
# Up to but not including the number on the right.
#  [0,1,2,3]
print(list(oneThroughFive))
print(list(zeroToFour))


print(list(range(11)))
# Loop through a range 0 to 4:
for index in range(0, 5):
    print(index)


# Loop through a list of names:
people = [ "Bob", "Carol", "Ted", "Alice" ]

for person in people:
    print( person )

#dict
fruit = {"A":"apple","B":"banana","C": "cherry"}

#loops through just the keys
for f in fruit:
    print( f )

# Loop until we reach zero:

countDownFrom = 10 # Set the initial value

# Enter the loop only if the expression is true.
while countDownFrom >= 1:
    print( countDownFrom )
    # Decrement the count
    countDownFrom = countDownFrom - 1

print("DONE")


age = 12

ingredients = ["coke"] # list

if age > 100:
    print( "May Not Drink." )
    print( "You have had a lifetime worth..." )    
elif age >= 21:
    ingredients.append("shot") # add to list
    print( "May Drink." )
    print( "Yay!" )
else:
    print( "May Not Drink." )
    print( "Darn..." )

print("more code")
print(ingredients)
def addTwo(a:int = 0, b:int = 0) -> int:
        return a + b

addResult1 = addTwo()
addResult2 = addTwo(2, 4)

print(addResult1)
print(addResult2)

def is_of_age(age): # receive a parameter
    return age >= 21

# *variable* 
a = 12
result = is_of_age(a) # passing an *argument*
print(result)

print(is_of_age(99))
def game_over():
    print("Game Over!")

game_over()

for n in range(5):
    game_over()
def full_name(first_name, last_name):
    return f"{last_name}, {first_name}"

print(full_name("Kevin", "Long"))
def twice(text: str) -> str:
    return f"It's {text} and {text} again!"

twiceResult = twice("abc")
print(twiceResult)

def get_smallest(numbers: [int]) -> int:
    smallest = None
    #PUT YOUR CODE HERE
    # What are the steps?
    #LOOP through list
    for  item in numbers:
        #if you see a new value that is lower or its the first item in the list (e.g.  == None)
        if smallest is None or item < smallest:
            smallest = item
    #then replace smallest with new smallest
    
    return smallest

#Sample Data
p = [123,597,631,61,93,509]

#Test function
print(get_smallest(p))
# EXTRA CREDIT largest, total, average
def total(numbers):
    subtotal = 0
    for n in numbers:
        subtotal += n
    return subtotal

print(total([1,2,3,4]))

print(sum([1,2,3,4]))
def average(numbers):
    subtotal = 0
    how_many = len(numbers)
    for n in numbers:
        subtotal += n
    return subtotal / how_many
print(average([123,597,631,61,93,509]))
text = "ABC"
first = text[0]
last = text[-1]
print(first,last)
print(list(text))

#print stack on top each on its own line
for letter in text:
    print(letter)

together = "X" + "Y"
together = together + "Z"
together += "Z"
print(together)

# print on seperate lines
print("KEVIN\nERNEST\nLONG")
def reverse_me(text):
    output = ""
    # TODO: Put Your Code Here
    for letter in text:
        output = letter + output
    return output

# TEST

result = reverse_me("ABC")
print(result)
assert("CBA" == result)

# JOIN AND SPLIT
# Join array of strings together using a strin comma as glue
print(",".join(["AAA","BBB","CCC"]))

print(" ".join(["AAA","BBB","CCC"]))

comma = ","
data_list = ["AAA","BBB","CCC"]
print(comma.join(data_list))

data_text = "AAA,BBB,CCC"
output = data_text.split(",")
print(output)
for item in output:
    print(item)

def reverse_me(text):
    output = []
    # TODO: Put Your Code Here
    index = len(text) - 1
    while index >= 0:
        output.append(text[index])
        index = index - 1
    return "".join(output)

# TEST

result = reverse_me("ABC")
print(result)
assert("CBA" == result)

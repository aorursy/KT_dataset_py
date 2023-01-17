a = 22 # Assignment
b = 33 # Assignment
x = a + b # Expression and Assignment
print( x ) # Print the Result
y = a * b # Expression and Assignment
print( y ) # Print the Result
y = a / b # Expression and Assignment
print( y ) # Print the Result
half: float	= 0.5
third = 1 / 3
print ( half )
print ( third )

example = "This is 1 example of a text string."
text = "ABC"
print(example)
print(text)
print(example + text) #CONCATENATE
quantity = 123
output = f"The quantity is { quantity }."
print(output)

print( "ABC" + str(987) + "XYZ")

price = 200
print(f"The total cost for { quantity } items at { price } each, is { quantity * price }")
third = 1 / 3
print(f"{third:.2}")
isCool = True
isTooCool = False

isGreater = 3 + 1 >= 4
isSame = 2 + 2 == 5 - 1

print(isGreater)
print(isSame)
# Array of integers
numbers = [ 12, 23, 54 ]
#numbers = list( [12, 23, 54] )

# Array of Strings
names = [
	"Bob", 
	"Carol", 
	"Ted", 
	"Alice"
]

print(numbers)
print(names)

print(numbers[2])
print(names[3])
print(names[-2])
print(names[2])
print(names[-1])
# Declare an Array of Strings
names: [str] = [] 
    
# Append two items
names.append( "Bob" )
names.append( "Carol" )
assert(len(names) == 2 ) # Verify

# Print Them Out
print( names[ 0 ] )
print( names[ 1 ] )

# Remove One Item
names.remove( "Bob" )
assert( len(names) == 1 ) # Verify

phrase_book: {str:str} = {
    "alpha" : "A as in Alpha",
    "bravo" : "B as in Bravo",
    "charlie" : "C is for Charlie",
}

phrase = phrase_book[ "bravo" ]
print( phrase )
assert("B as in Bravo" == phrase)


coin_quantities:{int:int} = {}

coin_quantities[ 1 ] = 5 # We have 5 pennies
coin_quantities[ 5 ] = 4 # We have 4 nickels
coin_quantities[ 10 ] = 3 # We have 3 dimes
coin_quantities[ 25 ] = 2 # We have 2 quarters

quarter_count = coin_quantities[ 25 ]
print(quarter_count)
assert( 2 == quarter_count )


print( coin_quantities[ 10 ] )
oneThroughFive = range(1,6)
# up to but not including the number on the right.
# [1,2,3,4,5]

zeroToFour = range(0,5) 
# Up to but not including the number on the right.
#  [0,1,2,3,4]
print(list(oneThroughFive))
print(list(zeroToFour))

books = {}
title1 = "Good Omens"
books[title1] = "Adventures in the Apocolypse."
print(books[title1])
# Loop through a range 0 to 4:
for index in range(5):
    print(index)

# Loop through a list of names:
people = [ "Bob", "Carol", "Ted", "Alice" ]

for person in people:
    print( person )

# Loop until we reach zero:

countDownFrom = 10 # Set the initial value

# Enter the loop only if the expression is true.
while countDownFrom > 0:
    print( countDownFrom )
    # Increment the count
    countDownFrom = countDownFrom - 1 

age = 6

low = 8
limit = 99

if age >= 8 and age <= 99:
    print( "May Play." )
else:
    if age > 99:
        print( "May Still Play ." )
    else: # Under 8
        print("Parental supervision required.")

print("Done!")
light = ""

if light == "green":
    print("go")
elif light == "red":
    print("stop")
elif light == "yellow":
    print("go very fast!")
else:
    print("treat as a four way stop")

data = [111, 222, 333, 444, 555]

# total = ???

data = [111, 222, 333, 444, 555]

# total = ???
# for item in data:

total = 0
for index in range(0, len(data)):
    total = total + data[index]
print(total)

total = 0
for item in data:
    total = total + item
print(total)

# total = data[0]
# total = total + data[1]
# total = total + data[2]
# total = total + data[3]
# total = total + data[4]

a = 2  #  Assignment
b = 3  #  Assignment

x = a + b  #  Expression and Assignment
y = a * b  #  Expression and Assignment

print( x )  #  Print the Result
print( y )  #  Print the Result

print(12 * 12)
example = "This is 1 example of a text string."
text = "ABC"
print(example)
quantity = 123
output = f"The quantity is { quantity }."
print(output)
#CONCATENATE
output = "ABC" + "XYZ"
print(output)

quantity = 123
print("quantity: " + str(quantity))
isCool = True
isTooCool = False

isGreater: bool = 3 < 2
isSame: bool = 2 + 2 == 5
    
print(isGreater)
print(isSame)

# Array of integers
numbers = [ 12, 23, 54 ]
print(numbers)

# Array of Strings
names = [
    "Bob", 
    "Carol", 
    "Ted", 
    "Alice"
]
print(names)
print('I can\'t even...')
print("<img src=\"file.png\">")
print("""
I can't even!!!
<img src="">
""")
# Array of integers
numbers = [ 12, 23, 54 ]

# Array of Strings
names = [
    "Bob", 
    "Carol", 
    "Ted", 
    "Alice"
]
print(numbers)
print(numbers[2]) #54
print(numbers[-1]) #54

print(names[-2]) #Ted
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

assert("B as in Bravo" == phrase)
print( phrase )

oneThroughFive = range(1,6)
# up to but not including the number on the right.
# [1,2,3,4,5]

zeroToFour = range(0,5) 
# Up to but not including the number on the right.
#  [0,1,2,3,4]
print(list(oneThroughFive))
print(list(zeroToFour))

print(list(range(10)))
data: [str] = [123, 456, "ABC", "XYZ"]
print(data)
age = 77

if age < 21:
    print( "May Not Drink." )
elif age > 50 and age < 70:
    print("Should not drink.")
elif age >= 70:
    print("Why not drink.")
else:
    print( "May Drink." )
    
print("All done")
# Loop through a range 1 to 4:
data = range(1,5)
for index in data:
    print(index)

# Loop through a list of names:
people = [ "Bob", "Carol", "Ted", "Alice" ]

people.remove("Ted")

print(people)

for x in people:
    print( x )

print("All done")


# for index in range(0,len(people)):
#     print(people[index])
count = 10
while count >= 0:
    print(count)
    count = count - 1
#     count -= 1 #shorthand
#     count-- # not in python
print("Blastoff!")
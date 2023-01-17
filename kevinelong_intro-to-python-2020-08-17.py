a = 123
print(a)

b = 456
print(a + b)
a = 2			# Assignment
b = 3			# Assignment

x = a + b		# Expression and Assignment
y = a * b		# Expression and Assignment

print( x )		# Print the Result
print( y ) 		# Print the Result

half: float	= 0.5
third: float = 1.0 / 3.0
print ( half )
print ( third )

# BOOLs are either True or False.
isCool = True
isTooCool = False

# Booleans are often created from other types using the comparison operators: 
# Equals "==" Greater Than ">"
# Less Than "<" and Not Equal "!="
isGreater: bool = 3 < 2
isSame: bool = 2 + 2 != 4

print(isCool, isTooCool, isGreater, isSame)
# Types may be declared before assigning a value. 
the_name: str
the_number: int
the_fraction: float
the_result: bool

# Or, they may be explicitly declared while also assigning a value.
another_name: str = "John"
another_number: int = 21
another_fraction: float = 3.14159
anotherResult: bool = True
print(another_name)
# Array of integers
numbers = [ 12, 23, 54 ]

# Array of Strings
names = [
    "Bob", 
    "Carol", 
    "Ted", 
    "Alice"
]

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

coin_quantities:{int:int} = {}

coin_quantities[ 1 ] = 5 # We have 5 pennies
coin_quantities[ 5 ] = 4 # We have 4 nickels
coin_quantities[ 10 ] = 3 # We have 3 dimes
coin_quantities[ 25 ] = 2 # We have 2 quarters

quarter_count = coin_quantities[ 25 ]

assert( 2 == quarter_count )

oneThroughFive = range(1,6)
# up to but not including the number on the right.
# [1,2,3,4,5]

zeroToFour = range(0,5) 
# Up to but not including the number on the right.
#  [0,1,2,3,4]
print(list(oneThroughFive))
print(list(zeroToFour))

# Loop through a range 0 to 4:
for index in range(0, 5):
    print(index)

# Loop through a list of names:
people = [ "Bob", "Carol", "Ted", "Alice" ]

for person in people:
    print( person )

# Loop until we reach zero:

countDownFrom = 10 # Set the initial value

# Enter the loop only if the expression is true.
while countDownFrom >= 1:
    print( countDownFrom )
    # Increment the count
    countDownFrom = countDownFrom - 1 

age = 11

if age < 21:
    print( "May Not Drink." )
elif age >= 50:
    print( "Should Not Drink." )
else:
    print( "May Drink." )


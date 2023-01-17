names = [
    "Bob",
    "Carol",
    "Ted",
    "Alice",
]

item_count = len(names)  #  LENGTH/COUNT of a list or string
print(item_count)

print(len([11,22,33]))  #  ONE LONGER THAN THE LAST INDEX
coin_quantities = {}  #  CREATE A DICT OBJECT WITH {} AND USE  OPTIONAL TYPE HINTING: :{int:int}

coin_quantities[ 1  ] = 5   #  We have 5 pennies
coin_quantities[ 5  ] = 4   #  We have 4 nickels
coin_quantities[ 10 ] = 3  #  We have 3 dimes
coin_quantities[ 25 ] = 2  #  We have 2 quarters

print(coin_quantities)

quarter_count = coin_quantities[ 25 ] # USE KEY OF 25 to access quarter count value of 2

print(quarter_count)

assert( 2 == quarter_count ) #TEST

## A RANGE CREATES A SEQUENCE OF NUMBERS THAT CAN BE EVALUATED LIKE; OR CONVERTED TO; A LIST

oneThroughFive = range(1,6)
# up to but not including the number on the right.
# [1,2,3,4,5]
print(list(oneThroughFive))

zeroToFour = range(0,5) 
# Up to but not including the number on the right.
#  [0,1,2,3,4]
print(list(zeroToFour))

zeroToFour = range(5) 
# Up to but not including the number on the right.
#  [0,1,2,3,4]
print(list(zeroToFour))

# Loop through a range 0 to 4:
for index in range(0, 5):
    print(index)



people_list = [ "Bob", "Carol", "Ted", "Alice" ]
count = len(people)
for index in range(count):
    item = people_list[index]
    print(item)
# Loop through a list of names:
people_list = [ "Bob", "Carol", "Ted", "Alice" ]
for item in people_list:
    print( item )  
print("DONE WITH LIST")
# Loop until we reach zero:
countDownFrom = 10 # Set the initial value
# Enter the loop only if the expression is true.
while countDownFrom >= 1:
    print( countDownFrom )
    # Increment the count
    countDownFrom = countDownFrom - 1
print("BLAST OFF")
age = 20

if age < 21: #Expression
    print( "May Not Drink." ) # if expression evaluates to True
elif age > 88: #else if is abbreviated to elif
    print( "Should Not Drink." ) # if expression evaluates to True
else: #default - no expression
    print( "May Drink." ) #if both expressions above evaluate to False


def addTwo(a:int = 0, b:int = 0) -> int:
        return a + b
addResult1 = addTwo()
print(addResult1)
def addTwo(a, b):
        return a + b
addResult2 = addTwo(2, 4)
print(addResult2)
def addTwo(a, b):
        return a + b

print(addTwo(2, 4))
def twice(text):
    return f"{ text }{ text }"

twiceResult = twice("abc")
print(twiceResult)
assert(twiceResult == "abcabc")
numbers = [11,55,33,66,22]
for n in numbers:
    print(n)
n = 49
if n >= 50:
    print(n)
else:
    print("too small")
numbers = [11,55,33,66,22]
for n in numbers:
    if n >= 50:
        print(n)
    else:
        print("too small")    

def do_work(numbers):
    for n in numbers:
        if n >= 50:
            print(n)
        else:
            print("too small")  


do_work([11,55,33,66,22])

do_work([19,955,393,6,22])

numbers = [11,55,33,66,22]

# total a list

# lowest in list

# highest in list

# average list


# total a list
numbers = [11,55,33,66,22]
total = 0
for n in numbers:
#     total += n
    total = total + n
print(total)
# total a list AS A FUNCTION
def total(numbers):
    subtotal = 0
    for n in numbers:
        #total = total + n
        subtotal += n
    return subtotal
    
print(total([11,55,33,66,22]))
# LOWEST
numbers = [11,55,9,66,22]

lowest = numbers[0]

for n in numbers:
    if n < lowest:
        lowest = n
print(lowest)

# LOWEST

def get_lowest(numbers):
    lowest = numbers[0]

    for n in numbers:
        if n < lowest:
            lowest = n
    return lowest

print(get_lowest([11,55,9,66,22]))

# AVERAGE

def get_average(numbers):
    subtotal = 0

    for n in numbers:
        subtotal = subtotal + n
    
    count = len(numbers)
    
    return subtotal / count

print(get_average([11,55,9,66,22]))
def make_list(a,b,c):
    return [a,b,c]

print(make_list(9,8,7))
def get_currency_text(amount):
    return f"${amount:.2f}"

print(get_currency_text(12.345))
def addTwo(a, b):
    return a + b

print(addTwo(3,4) * 20)

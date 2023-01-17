print("Hello, World!")
print("Hello, World!")
a = 5

print(a) # Integer



b = 5.5

print(b) # Float



c = "Hello!"

print(c) # String
"""Comment

Lines"""

print(a)
a = 5

A = 6

print(a,A)
print("Type of a:", type(a))

print("Type of b:", type(b))



a = str(a)

b = int(b)



print("Type of a:", type(a))

print("Type of b:", type(b))
a = 5

b = 6

print(a+b)
a = 5

b = 3



print("a + b =", a + b)

print("a - b =", a - b)

print("a * b =", a * b)

print("a / b =", a / b)

print("a % b =", a % b)

print("a ** b =", a ** b)
a = 5

b = 7



a += b # a = a + b

b **= 2 # b = b**2



print(a)

print(b)
a = 5

b = 5



print(a == b) # a equal to b

print(a >= b) # a equal to b or greater than b

print(a != b) # a not equal to b
a = 6

b = 6

c = 2

print(a == b or c == 5)

print(a == b and c == 3)
print("alp" in "alpha")

print("b" in "alpha")
listA = ["Alpha", "Gamma", "Delta"]

print(listA)
print(listA[0])
listA[0] = "Omega"

print(listA)
print(len(listA))
listA.append("Alpha")

print(listA)
listA.remove("Delta")

print(listA)
listA.sort()

print(listA)
print(listA)

print(listA[0:2])
print(listA[1:])
a = "Hello, World!"

print(type(a))

print(a[0])

print(len(a))
print(a.lower())

print(a.upper())
print(a.replace('World', 'Sekai'))
print(a[0:9])

print(a[2:11])
part1, part2 = a.split(',')

print(part1)

print(part2)
print(a.split('o'))

x, y, z = a.split('o')

print(x)

print(y)

print(z)
var = "Narita"

print("We are in "+var+" city.")
tupleA = ("Alpha", "Bravo", "Charlie")

print(tupleA)
print(tupleA[1])
setA = {"Quadra", "Double", "Triple"}

print(setA)
setA.add("Hexa")

print(setA)
dictA = {

    "Name": "The Hitchhiker's Guide to the Galaxy",

    "Author": "Douglas Adams",

    "Language": "English"

}

print(dictA["Language"])
dictA["Language"] = "Japanese"

print(dictA)
dictA["Genre"] = "Science Fiction"

print(dictA)
rain_sensor = 0



if (rain_sensor == 1):  # The "rain_sensor == 1" statement must be True to drawing curtains.  | BLOCK 1

    motors = "Half Power"                                                                  #| BLOCK 1

    print("Raining")                                                                       #| BLOCK 1

    

else:                   # If "rain_sensor == 1" not true this block will work.                | BLOCK 2

    print("Not Raining")                                                                   #| BLOCK 2
Electric = 0

Benzin = 1



if Electric == 0 and Benzin == 1:

    Generators = 1

    print("Generators starting...")
mode = "Semi Automatic"



if mode == "Full Automatic":

    print("System Set to Full Automatic Mode")

elif mode == "Semi Automatic":

    print("System Set to Semi Automatic Mode")

else:

    print("System Set to Manuel Mode")
mode = "Semi Automatic"



if mode == "Full Automatic":

    print("System Set to Full Automatic Mode")

    if mode == "Semi Automatic":

        print("System Set to Semi Automatic Mode")

else:

    print("System Set to Manuel Mode")
mode = "Semi Automatic"



if "Automatic" in mode:

    if mode == "Full Automatic":

        print("System Set to Full Automatic Mode")

    if mode == "Semi Automatic":

        print("System Set to Semi Automatic Mode")

else:

    print("System Set to Manuel Mode")
counter = 0



while (counter < 3):

    print(counter)

    counter = counter + 1
counter = 0



while counter < 10:

    if counter % 2 == 0:

        print(counter)

    counter = counter + 1
print(list(range(10)))
print(range(10))

print(type(range(10)))
for i in range(3):

    print(i)
print(list(range(6,14,3)))
listA = list(range(6,14,3))



print("List : ", listA)



print("")

print("For 1")



for i in listA:

    print(i)



print("")

print("For 2")



for i in range(6,14,3):

    print(i)
listA = ["Uniform", "Delta", "Sierra", "Oscar"]



for i in listA:

    print("--> "+i)
print(listA)

print(len(listA))

print(range(len(listA)))

print("-------------------")



for i in range(len(listA)):

    print(i)
for index, i in enumerate(listA):

    print("index =", index, "/ iterator =", i)
for i in range(5):

    if i == 3:

        break

    print(i)
for i in range(5):

    if i == 3:

        continue

    print(i)
def square(number):

    square_of_number = number * number

    return square_of_number
a = 7

result = square(a)



print(result)
print(square(11))
def divider(numberA, numberB):

    

    if numberB == 0:

        print("Divisor can't be zero!")

        return 0, 0

    

    quotient = int(numberA / numberB)

    remainder = numberA % numberB

    

    return quotient, remainder
print(divider(17, 5))
quotient, remainder = divider(21, 10)

print("Quotient =", quotient, "Remainder =", remainder)
quotient, remainder = divider(6, 0)

print("Quotient =", quotient, "Remainder =", remainder)
mod = lambda num1, num2 : num1 % num2

print(mod(11, 8))
listA = [1, 2, 3, 4, 5]

squares = []

for i in listA:

    squares.append(i**2)

    

print(squares)
print(list(map(lambda x: x**2, list(range(1,6)))))
import math



print(math.sin(math.radians(30)))
listA = list(range(100))



new_list = []



for i in listA:

    if i % 2 != 0:

        new_list.append(i**2)



print(new_list)
listA = list(range(100))



print([i**2 for i in listA if i % 2 != 0])



# One-Line = print([i**2 for i in list(range(100)) if i % 2 != 0])
import os



os.listdir("../input/") 
os.mkdir("./test")

os.listdir(".")
os.rmdir("./test")

os.listdir(".")
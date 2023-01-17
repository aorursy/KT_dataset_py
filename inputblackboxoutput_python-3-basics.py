# This is a single line comment

# This is a
# multi-line
# comment 

'''This is a docstring (Used to document stuff)'''
# This program does nothing!
def func():
	pass
func()
# Importing modules
import math

# Importing modules with alias
import pickle as p

# Importing specific sections from a module
from math import sqrt
math.pi
# math.sqrt(4)
sqrt(4)
# Printing to the console
print("String Text")                 # Newline
print("String", end="")              # No newline

# Printing in a formatted manner
name = 'Rutuparn Pawar'
age = 20
print("My name is: {}, and my age is: {}".format(name, age))   # Using .format
print(f"My name is: {name}, and my age is: {age}")             # Using fstring


# Getting input from the user
variable1 = int(input("Enter a number:"))                                         #Single integer variable
print(variable1)

variable1, variable2 = input("Enter 2 number (with space in between):").split()   #Multiple variable
print(variable1, variable2)
age = 20                       # Integer
pi = 3.142                     # Floating point number
name = 'InputBlackBoxOutput'   # String
isClosed = True                # Boolean  
text = "Python is fun!"
speech1 = "Mark said, \"Python is fun \" "
speech2 = 'Mark said, "Python is fun"'

escChar = 'a \n b \t c \\ d \" e \' f'   # Only a few shown 

# Useful string functions
print(text.lower())
print(text.upper())
print(text.split())  # By default split works with space

# String concatenation
str1 = 'My name is '
str2 = 'Rutuparn Pawar'
print(str1+str2)

# Getting char from string using bracket notation
print(text[0])     # 'P' printed on screen
a = 10
b = 5
print("Addition = ", a+b)         # Addition
print("Substraction= ",(a-b))     # Substraction
print("Division= ",(a/b))         # Division
print("Multiplication= ",(a*b))   # Multiplication
print("Remainder= ",(a%b))        # Remainder operator
print("Power=",(a**b))             # Power
print(a==b)
print(a!=b)
print(a>b)
print(a>=b)
print(a<b)
print(a<=b)
print(a and b)
print(a or b)
print(not True)
a = 2
b = 4

# if
if a>b:
	print(f'{a} is greater than {b}')

# if else
if a>b:
	print(f'{a} is greater than {b}')
else:
	print(f'{b} is greater than {a}')

#if elif else
if a>b:
	print(f'{a} is greater than {b}')
elif a==b:
	print(f'{a} is equal to {b}')
else:
	print(f'{b} is greater than {a}')
# for with range
for x in range(0, 5):
	print(x)

# for with list
for each in ['one', 'two', 'three']:
	print(x)

# while
i=0
while(i<10):
	print(i)
	i+=1
lis1 = [1,2,3,4]                       # Same datatypes
lis2 = [1,True,'InputBlackBoxOutput']  # Different datatypes
lis3 = list(range(0,100))              # List with values 0 to 99

# Adding to the end of a list
print(lis1.append(5))    

# Removing from the end of the list
print(lis1.pop())

#Accessing an element from the list
print(lis2[2])          # 3rd element = 'InputBlackBoxOutput'

# List comprehension
lis = [x**3 for x in range(0,4)]    # lis = [0, 1, 8, 27]



# Nested list
nestedList = [[1,2,3], [4,5,6], [7,8,9]]

#Accessing an element from the list
print(nestedList[0][2])          # 1st list 3rd element = 'InputBlackBoxOutput'
t = (1,2,3)

# Changing a tuple is not possible!
# t[0] = 20 

# Accessing value from a tuple
print(t[0])
d = {'a':1, 'b':True, 'c':'String'}

# Changing dictionary value associated with a key
d['b'] = False

# Accessing dictionary value for a key
print(d['c'])

# Getting dictionary keys
print('Keys = ', d.keys())
print('Values =', d.values())
# import math
def vectorMagnitude(x, y, z):
	return math.sqrt(x**2 + y**2 + z**2)

print(vectorMagnitude(10, 10, 10))     # O/P = 17.32

# Function with default parameter
def greetUser(name='User'):
	print('Hello ' + name)

greetUser()             # O/P => Hello User
greetUser('Rutuparn')   # O/P => Hello Rutuparn




# Lamda expressions (AKA anonymous function)
# Syntax => something = lambda input:output
square = lambda x: x**2

print(square(10))
# (Please see lambda functions before viewing this section)
test = [1, 2, 3, 4]

# Apply to all elements
test = list(map(lambda x:x*100, test))      # Returns list with all elements multiplied by 100

# Apply condition on a list
test = list(filter(lambda x:x>=3, test))    # Returns list element >= 3 

# Find a computation using all list elements
from functools import reduce
test = reduce(lambda x,y:x+y, test)         # Returns sum of list elements

test= [1,1,2,2,3,4,8,5]
# Find unique values 
set(test)
 
def func(*args):                  # Tuple
    for each in args:
        print(each)

func(1, "Apple", True)


def kw_func(**kwargs):            # Dictonary
    print(kwargs)
    for each in kwargs.keys():
        print(kwargs[each])
        
kw_func(a=10, b="Cool")
try:
    print(1/0)  # Division by zero!
except:
    print("Error occured")
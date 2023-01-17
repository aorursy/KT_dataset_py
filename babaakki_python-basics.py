25*4

#Benifits of interactive mode: Can test bits/chunks of code beforehand.
help('print')
print('Hello World!\nSo typical.')
type('Hi')
type(17)
type(25.4)
type(True)
#Variable Declaration and initialization

string = "Have a good day." #Single quotes and Double quotes both are accpetable for string

n = 25

pi = 3.14

truth_value = True
#Accessing the variable

pi
#Arithmetic Operators

# +  Addition

# -  Subtraction

# *  Multiplication

# /  Division
#Exponentiation

5**3
#Normal Division

59/60
#Floor Division

61//60
'String ' + 'Concatenation'
'String' * 4
### Boolean/Comparison Operation

# ==  Comparator "Equals to"

# !=  "Not equals to"

# >  "Greater than"

# <  "Less than"

# <=  "Greater than equals to"

# >=  "Less than equals to"
100%2 == 0
### Logical operators

# and

# or

# not   negation
a, b, c = 2, 3, 5

not((a>b and a>c) or (a+b == c))
print('c' in 'miracle')
print('a' is 'A')
#To check the type of a variable

real = 25.4

type(real)
#To print

print("This is print function.")
#Type Conversion

str(25)
int('25')
bool(100)
#Int conversion does not rounds up

int(25.4), int(19.9)
import math
print(math)
math.sqrt(4.0)
#Alternate way to import a module

import math as m
m.pow(25, 4)
#Pre-defined constant in math module

m.pi
from math import pow

print(pow(2,5))
#importing everything from module

from math import *
cos(90/360)
sin(90/360)
def greetings(name):

    print('Hello '+name)
greetings('Mr. President')
#Function definition creates a variable with the same name of type function

type(greetings)
#Keyword arguement "n" initialised with a default value

#Unlike C functions, Python functions can return multiple values, sing Tuples ( discussed in the later section)

def operations(m, n=1):

    return m*n, m/n
m, d = operations(10, 2)

print(m, d)
operations(10)
def area_rectangle(length=10, bredth=50):

    #Docstring

    """

    A docstring is a string at the beggining of a function, explaining the function.

    This function takes 2 paramenter length and bredth and calculate the area of rectangle.

    """

    return length*bredth
print("Area : ", area_rectangle(100, 20))
print("Area for length {} and bredth {}: {}".format(100, 20, area_rectangle(100, 20)))
x = 100

if x % 10 == 0:

    pass  # Equivalent to do nothing

#Must have atleast one statement inside the block
x = 0

if x > 0:

    print("Positive")

elif x < 0:

    print("Negitive")

else:

    pass
import time  #built in module for time



#Make sure to include base condition, otherwise the function will call itself infinitely.

def countdown(n):

    #Base condition

    if n <= 0:

        print("Go Go Go...!")

    else:

        print(n)

        time.sleep(1)

        countdown(n-1)

        

countdown(3)
num = input("Enter a number: ")
type(num)
#By default type of the entered input is string

#Type-cast is needed

num = int(input("Enter a number: "))
#int() requires input string must contains digits only, if not then, it will prompt runtime error

type(num)
try:

    num = int(input("Enter a number between 0 and 9:\t")) # \t is a special character for tab-space

    if num % 2 == 0:

        print("{} is Even".format(num))

    else:

        print("{} is Odd".format(num))

except Exception as e:

    print(e)
c = 10
c = c + 10

c
c += 10  #Verify for other operators

#Another way to write formatted print statement

print("Result is %d" % (c))
n = 10

while(n != 0):

    print(n)

    n -= 1

print("Done..!")
#break: jump out of the loop

n = 10

while(n != 0):

    if n >= 5:

        print(n)

    else:

        break

    n -= 1

print("Done..!")
#continue: to skip the following code for a particular iteration

n = 10

while(n != 0):

    if n%3 == 0:

        n -= 1

        continue

    print(n)

    n -= 1

print("Done..!")



#Try commenting n-=1 inside the if block; And see what happens
#pass: Do Nothing

n = 10

while(n != 0):

    if n%3 == 0:

        pass

    else:

        print(n)

    n -= 1

print("Done..!")



#Compare with execution of continue.
# Use help('range') to know more about the function

for i in range(10):

    print(i)
greetings = "Hello Mr. President"
#multiline string

#\n --> new line

multi = """ Hello, How are you?

Hi, I am fine. Thank You."""

multi
#Accessing an element at any location. Index starts from 0 and goes upto length-1

#Any exprsession or variable, resulting into integer can be used as an index

greetings[4]
#Negitive Indexes

greetings[-1]
#len() built-in function

len(greetings)
prefixes = 'JKLMNOP'

suffix = "ack"

for char in prefixes:

    print(char+suffix)
#[n:m] :: nth character included and mth character excluded

greetings[3:8]
greetings[:8]
greetings[3:]
#third arguement is step size

greetings[0:10:2]

#Check what negitive stepsize can do
#greetings[0] = 'F'
string = 'TaDaa'
string.lower()
string.upper()
string.capitalize()
string.find('a')
string.replace('a', 'o')
#Removes extra space from left and right

#check out for rstrip and lstrip

'  Hello. '.strip()
student = ['Rohan', 20, 8.5, True, ['Football', 'Table Tennis']]

student
#Empty List

[]
#Another way to create. using built-in function

l = list()

type(l)
#List element are accessing is similar to accessing characters in string

student[3]
len(student)
student[1] = 21

student
for elem in student:

    print(elem)
#Concatenation

[1,2,3]+[4,5,6]
#Repetition

[1,2,3]*5
lst = [1,2,3]*5

lst[4:9]
lst[8:]
lst[:8]
#Adding an element to the end of the list

student.append(['Python', 'C++'])

student
#Appending all the elements of another list

student.extend([180, 85])

student
#Inplace sorting

#sort arranges in ascending order

name = ['j', 'o', 'h', 'n']

name.sort()

name

#There is another method 'sorted'. Explore!

#Check what happens, name = name.sort()
[elem for elem in lst if elem % 2 != 0]
#Deleting using the index and returns the element deleted

ele_pop = student.pop(-3)

student, ele_pop
#Using del

del lst[3]

lst
del lst[3:]

lst
#returns None

student.remove(True)

student
#string in list of character

list(greetings)
#String to list of words

#can also split on any character/special-symbol, check documentation

greetings.split()
#joing list of strings into a sentence

#string method

' & '.join(student[3])
#Empty Dictionary

eng2hnd = {}

eng2hnd
type(eng2hnd)
#key:value

#Every element in the dictionary is inthe form of key-value pair

eng2hnd = {'one':'ek', 'two':'do', 'three':'teen', 'four':'chaar'}

eng2hnd
#adding a new item in the dictionary

eng2hnd['five'] = 'paanch'

eng2hnd
#Another way to create. using built-in function

d = dict()

type(d)
len(eng2hnd)
#To check whether something is key in a dictionary keys or not

'one' in eng2hnd
#To check whether something is key in a dictionary values or not

'uno' in eng2hnd.values()
for key in eng2hnd:

    print(key, eng2hnd[key])
for key, val in eng2hnd.items():

    print(key, val)
tup = ('a', 'b', 'c', 'd')

tup
type(tup)
#Not necesasry to enclose in a paranthesis

tup = 'a', 'b', 'c', 'd'

tup
#Another way to create. using built-in function

t = tuple()

type(t)
#Accessing values of tuple

tup[2]
len(tup)
tup[1:3]
tup[:2]
tup[1:]
tup = tup + ('e', 'f', 'g')

tup
tup * 2
#Following from function

type(operations(10, 2))
#built-in function returning multiple values

divmod(25,4)
#enumerate: travels elemnts and its indices together.

#For every item returns tuple of 2 item, index and the element itself

lst = list(enumerate('abcde'))

lst
dct = dict(lst)

dct
#open a file

#open(filename/filepath, mode)

#r - read

#w - write

#Make sure the file is present in the given filepath

file_object = open('../pythonFiles/script.py', 'r')

file_object
#Reads the entire file and return the result as a string

file_object.read()
#reads the file until it encounters newline character and returns the result as a string

file_object.readline()
file_object.close()
#If file already exists, opening in the write mode clears all the previous content.

#If the file doesn't exist, creates a new one.

#The arguement of write has to be string

fobj = open('hello.txt', 'w')

fobj.write("Hello\n")

fobj.write("Welcome to area %d.\n" % (51))

fobj.close()
#explore other modes

help('open')
#Another way to open a file using with keyword.

#File automatically gets closed, once you come out of the with block

with open('hello.txt', 'r') as fobj:

    for line in fobj:

        print(line.strip())
import os
#returns path of the current working directory

cwd = os.getcwd()

print(cwd)
#get the absolute path of a file

os.path.abspath('hello.txt')
#check whether file or directory exists

os.path.exists('script.py')
#checks whether it's directory

#'isfile' checks whether file or not

os.path.isdir('./../CS242')
os.listdir()
path = os.path.join(cwd, 'script.py')

path
#to execute commands from the python program

# check this link  https://docs.python.org/3/library/os.html#os.system

os.system('ls -la > ls.txt')
#For command-line arguement

#Check file cmd.py

!python cmd.py roger that 
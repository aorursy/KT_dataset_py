def jorge_the_goose():    # Define a function that takes no arguments, and returns no values
    print("Honk")
    print("bite")

jorge_the_goose()

print("Here's a loop calling our function:")
for i in range(3):
    jorge_the_goose()
def add(a,b): # a and b are 'arguments'
    return a+b

print(add(2,4))
print(add(1.,3.2))
print(add(4,3.))
def double_and_halve(value):
    return value * 2., value / 2. # The comma separates the two values returned
                                  # These are returned as a `tuple` – list whose length and values cannot be changed (immutable)

print(double_and_halve(9))
d, h = double_and_halve(5.) # In this case, we capture the two outputs in two separate variables using a comma

print(d)

print(h)
def do_a():
    print("doing A")
    
def do_b():
    print("doing B")
    
def do_a_and_b():
    do_a()
    do_b()
    
do_a_and_b()
def rev_print(string):
    if len(string)>0:              
        print(string[-1], end='')  # Prints the last character. The 2nd argument prevents it from making a new line for each print
        rev_print(string[:-1])      # Calls itself, but on a string that is shorter by one
    else:
        return()                   # Stops the recursive loop if the string is empty

rev_print("Banana backwards is ananaB.")
# YOUR CODE HERE
def factorial(num):
    out = 1                    # start with 1
    for i in range(1,num+1):   # loop from 1 to your input number
        out = out * i          # multiply your starting by the loop number 
                               # It will execute 1 * 1 * 2 * 3 * ...
    return out

print("loop output: " + str(factorial(5)))

def recursive_factorial(num):
    if num > 1:
        return num * recursive_factorial(num-1)  # 5! = 5 * 4!, so I can call this function again on num-1
    else:
        return 1     # 1! = 1. If I don't have a final value, the whole thing will fail.

print("recursive output: " + str(recursive_factorial(5)))
d = {'a':1, 'b':2, 'c':3}
d['b']
d = {'a':1, 'b':2, 'c':3} # Redefining d just in case cells get executed out of order
print(d)

d['d'] = 4     # add a new key-->value pair
print(d)
if "b" in d:
    print("It's here")
else:
    print("Go fish")
# Kaggle puts this at the top of every new notebook
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
f = open('/kaggle/input/demodata/fruits.txt', 'r') # Imports the relevant file in read mode using 'r'
my_data = f.read() # Reads the file as a string, and saves the whole thing as a string

my_data
lines = my_data.split('\n') # Makes a list of lines divided by \n

lines
lines[0].split()  # Splits up the first line of lines
# Let's re-read the data just in case

f = open('/kaggle/input/demodata/fruits.txt', 'r') # Imports the relevant file in read mode using 'r'
raw_data = f.read()                                # Reads the file as a string, and saves the whole thing as a string
lines = raw_data.split('\n')                       # Makes a list of lines divided by \n

fruits = {}  # An empty list to hold our final dataset

# YOUR CODE HERE

for line in lines:
    columns = line.split()
    fruits[columns[0]] = columns[1]

print(fruits)
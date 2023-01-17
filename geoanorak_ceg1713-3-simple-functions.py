# Writing functions (part 1 simple functions)
#Remember this from last time.  This is the surface area of a pyramid equation
#Let's solve it in Python

import math #for sqrt and pow functions
r = 12.2  #radius
l = 9.71  #length (this is the letter "l" not the value 1!)
h = 6.591 #height
w = 7.25  #width

# This is relatively complex so we won't try and do it all in one line but break it up into smaller chunks
# We can assign the chunks variables and then complete the equation at the end

# Let's start from the right

# and inside to out
sa1 = math.pow(l/2, 2) + math.pow(h,2)
sa2 = w * math.sqrt(sa1)

# and the next chunk
sa3 = math.pow(w/2, 2) + math.pow(h, 2)
sa4 = l * math.sqrt(sa3)

#And finally

sa_pyr = l * w + sa4 + sa2
print(sa_pyr)
# And we should format our answer nicely as well
print("The surface are of a pyramid with length {}, width {} and height {} is {:.3f}".format(l,w,h,sa_pyr))

# Format is really powerful see https://mkaz.blog/code/python-string-format-cookbook/
# This is quite a complicated equation spread over many lines
# If we wanted to reuse it then our code becomes very repetitive
# In this case we might turn it into a function 
# Functions are reusable bits of code that spit out an answer (or sometimes multiple answers)
# They are FUNDAMENTAL to writing good scripts and codes

a = math.sqrt(100)
print(a)

b = math.pow(3, 3)
print(b)
# These 2 functions - they return values
# Note the first one has a single piece of information passed to it
# the second has 2 pieces of information passed to it
# We use the word "pass" when talking about data sent to functions
# This data is called the argument (or arguments)
# We use the terminology "call a function" when some piece of code uses a function
# The 2 key things are the TYPE of information and the order they are passed

a = math.sqrt('hello') # fails
power = 2
number = 3

math.pow(number, power)
math.pow(power, number) #names don't matter only order - 1st is the number, 2nd is the power to raise it by

#Internally functions assign their own names to any data passed to them
# Function to cube a number

def cube(x):
    c = x * x * x
    return c

#Indented portion is the function.
#Note the Colon
#Return keyword passes the answer back to the calling code
#We can now use this like any other function
print(cube(3))
print(cube(5))
c = cube(1.78)
print (c)
y = 20
res = cube(y) #name of variable changed inside function (only inside the function)
print(res)
# ??

def onesqrt(x):
    res = 1/math.sqrt(x)
    return res

print(onesqrt(100))
print(onesqrt(245.987))
# a function to calculate the distance between 2 coordinates
# using pythagoras theorum - the square of the hypotenuse is equal to the square of the sum of the 2 sides

def euclid(x1, y1, x2, y2):
    dx = x2 - x1 #change in x
    dy = y2 - y1 #change in y
    d = math.pow(dx, 2) + math.pow(dy, 2)
    d = math.sqrt(d)
    return d
dist = euclid(1, 1, 5, 5)
print(dist)

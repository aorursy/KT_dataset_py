print("You've successfully run some Python code")

print("Congratulations!")

print("Hello My name is Robee")
from learntools.core import binder; binder.bind(globals())

from learntools.python.ex1 import *

print("Setup complete! You're ready to start question 0.")
# create a variable called color with an appropriate value on the line below

# (Remember, strings in Python must be enclosed in 'single' or "double" quotes)

color = 'blue'

q0.check()
q0.hint()

q0.solution()
pi = 3.14159 # approximate

diameter = 3



# Create a variable called 'radius' equal to half the diameter

radius = diameter/2

# Create a variable called 'area', using the formula for the area of a circle: pi times the radius squared

area = pi*radius*radius

q1.check()
# Uncomment and run the lines below if you need help.

#q1.hint()

#q1.solution()
########### Setup code - don't touch this part ######################

# If you're curious, these are examples of lists. We'll talk about 

# them in depth a few lessons from now. For now, just know that they're

# yet another type of Python object, like int or float.

a = [1, 2, 3]

b = [3, 2, 1]

q2.store_original_ids()

######################################################################



# Your code goes here. Swap the values to which a and b refer.

# If you get stuck, you can always uncomment one or both of the lines in

# the next cell for a hint, or to peek at the solution.

c = a

a = b

b = c

######################################################################

q2.check()
a = [1, 2, 3]

b = [3, 2, 1]

a,b = b,a

print (a)

print (b)
#q2.hint()
#q2.solution()
(5 - 3) // 2
q3.a.hint()
#q3.a.solution()
(8 - 3) * (2 - (1 + 1))
#q3.b.hint()
#q3.b.solution()
a = 121

b = 77

c = 109



s = a+b+c

d = s//3

print(s,d)
# Variables representing the number of candies collected by alice, bob, and carol

alice_candies = 121

bob_candies = 77

carol_candies = 109



# Your code goes here! Replace the right-hand side of this assignment with an expression

# involving alice_candies, bob_candies, and carol_candies

Total = alice_candies + bob_candies + carol_candies

Candles_Distributed_Evenly = Total//3

Total_Distribution = Candles_Distributed_Evenly * 3



to_smash = Total - Total_Distribution



q4.check()
#q4.hint()

#q4.solution()
print("You've successfully started the Python code")
print("Congratulations!")
print("I have started successfuly my first exercise")
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex1 import *
print("Setup complete! You're ready to start question 0.")
# create a variable called color with an appropriate value on the line below
a="color"
color="blue"
# what is your favourite color?
print(color)
q0.check()
#q0.hint()
#q0.solution()
pi = 3.14159 # approximate
diameter = 3

# Create a variable called 'radius' equal to half the diameter
r='radius'
diameter=3
radius=diameter/2
print(radius)
# Create a variable called 'area', using the formula for the area of a circle: pi times the radius squared
pi=3.14159
pi=abs(pi)
a='area'
r=1.5
diameter=3
radius=diameter/2
area=pi * r*r
print(area)



# Check your answer
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
#reverse the integer of a and b
# create a temporary variable and swap values
temp=a
a=b
b=temp
print(format(a) and(format(b)))

print([a] and [b])
######################################################################

# Check your answer
q2.check()
print("You've successfully run some Python code")
print("Congratulations!")
print("Hello, world!")
# You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
# (But if you can understand that code, you'll probably find these lessons boring :)
from learntools.core import binder
binder.bind(globals())
from learntools.python.ex1 import *
print("Setup complete! You're ready to start question 0.")
# create a variable called color with an appropriate value on the line below.
color = "green"
q0.check()
q0.hint()
# create a variable called color with an appropriate value on the line below
# (Remember, strings in Python must be enclosed in 'single' or "double" quotes)
color = 'blue'
q0.check()
q0.solution()
pi = 3.14159 # approximate
diameter = 3

# Create a variable called 'radius' equal to half the diameter
radius = diameter / 2

# Create a variable called 'area', using the formula for the area of a circle: pi times the radius squared
area = pi * radius ** 2

q1.check()
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
a, b = b, a
######################################################################
q2.check()
(5 - 3) // 2
8 - 3 * 2 - (1 + 1)
q3.b.solution()
# Variables representing the number of candies collected by alice, bob, and carol
alice_candies = 121
bob_candies = 77
carol_candies = 109

# Your code goes here! Replace the right-hand side of this assignment with an expression
# involving alice_candies, bob_candies, and carol_candies
to_smash = (alice_candies + bob_candies + carol_candies) % 3

q4.check()
7------3
ninety_nine_dashes = 4
q5.check()
import random
from matplotlib import pyplot as plt
from learntools.python.quickdraw import random_category, sample_images_of_category, draw_images_on_subplots

## Step 1: Sample some sketches
# How many sketches to view - a random number from 2 to 20
n = random.randint(2, 20)
# Choose a random quickdraw category. (Check out https://quickdraw.withgoogle.com/data for an overview of categories)
category = random_category()
imgs = sample_images_of_category(n, category)

## Step 2: Choose the grid properties
######## Your changes should go here ###############
if n > 8:
    if n > 16:
        rows = 3
        cols = n // rows + 1
    else:
        rows = 2
        if n % 2 == 0:
            cols = n // rows
        else:
            cols = (n + 1) // rows
else:
    rows = 1
    cols = n

# The height and width of the whole grid, measured in inches.
height = rows * 2
width = cols * 2

## Step 3: Create the grid
grid = plt.subplots(rows, cols, figsize=(width, height))

## Step 4: Draw the sketches in the grid
draw_images_on_subplots(imgs, grid)
# The solution is kind of cool! However, it sometimes wastes space.
q6.solution()
# Solution
import random
from matplotlib import pyplot as plt
from learntools.python.quickdraw import random_category, sample_images_of_category, draw_images_on_subplots

n = random.randint(2, 20)
category = random_category()
imgs = sample_images_of_category(n, category)

rows = (n + 7) // 8
cols = min(n, 8)
height = rows * 2
width = cols * 2

## Step 3: Create the grid
grid = plt.subplots(rows, cols, figsize=(width, height))

## Step 4: Draw the sketches in the grid
draw_images_on_subplots(imgs, grid)
a = 0
b = 0
print(a, b)
q7.hint()
a = b = True
print(a, b)
a = b = 1.2
print(a, b)
a = b = "T"
print(a, b)
a = b = None
print(a, b)
q7.hint(2)
q7.solution() # I'm too young too naive.
# First Situation
odds = evens = []
for i in range(5):
    if (i % 2) == 0:
        evens.append(i)
    else:
        odds.append(i)
print(odds)
print(evens)
# Second Situation
L = [1, 2, 3]
a = b = L.pop()
print(a, b)
L = [1, 2, 3]
a = L.pop()
b = L.pop()
print(a, b)
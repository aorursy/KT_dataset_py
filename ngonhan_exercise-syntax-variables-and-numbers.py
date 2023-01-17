print("You've successfully run some Python code")
print("Congratulations! Mr Ngo")
print (1111)
print ('Oh my life')
# You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
# (But if you can understand that code, you'll probably find these lessons boring :)
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex1 import *
print("Setup complete! You're ready to start question 0.")
# create a variable called color with an appropriate value on the line below.
color = 'blue'
q0.check()
# create a variable called color with an appropriate value on the line below
# (Remember, strings in Python must be enclosed in 'single' or "double" quotes)
color = 'blue'
q0.check()
#q0.hint()
q0.solution()
pi = 3.14159 # approximate
diameter = 3

# Create a variable called 'radius' equal to half the diameter
radius = diameter / 2
# Create a variable called 'area', using the formula for the area of a circle: pi times the radius squared
area = pi * radius ** 2
q1.check()
# Uncomment and run the lines below if you need help.
q1.hint()
q1.solution()
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
a,b = b,a
######################################################################
q2.check()
#q2.hint()
#q2.solution()
(5 - 3) // 2
q3.a.hint()
#q3.a.solution()
8 - 3 * 2 - (1 + 1)
#q3.b.hint()
#q3.b.solution()
# Variables representing the number of candies collected by alice, bob, and carol
alice_candies = 121
bob_candies = 77
carol_candies = 109

# Your code goes here! Replace the right-hand side of this assignment with an expression
# involving alice_candies, bob_candies, and carol_candies
to_smash = (alice_candies + bob_candies + carol_candies) % 3 #-1

q4.check()
#q4.hint()
#q4.solution()
7------3
ninety_nine_dashes = 4
q5.check()
#q5.hint()
#q5.solution()
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
# The best couple of cols,rows is the least empty cells
# Number of Emptycells = (rows * cols) - n, with rows = (n // cols) + min (1, n% cols)
# so it become         = (((n // cols) + min (1, n% cols)) * col) - n
empty_cell = lambda col: ((((n // col) + min (1, n % col))* col) - n) 

# Bonus: Add balance,rather take 4x4 vs 8x2: append delta(row,col) = (abs((n // col) + min (1, n% col)-col)/10.0)
delta = lambda col: (abs((n // col) + min (1, n% col)-col))

# min evaluate with delta / 10 to avoid affect most empty condition
min_eval = lambda col: empty_cell(col) + delta(col)/10.0

# Use min with key as lambda min_eval
cols = min(8,7,6,5,4,3,2, key=min_eval)

rows = (n // cols) + min (1, n% cols)

# The height and width of the whole grid, measured in inches.
height = rows * 2
width = cols * 2

## Step 3: Create the grid
grid = plt.subplots(rows, cols, figsize=(width, height))

## Step 4: Draw the sketches in the grid
draw_images_on_subplots(imgs, grid)
#q6.hint()
#q6.solution()
#q7.hint(2)
#q7.solution()
# Variables point to values. 

# All variables have a value and a data type

# Strings can hold text, letters, and other characters

message = "Howdy, Everybody!"

print(message) # The print function prints whatever you put in the parentheses

message

# Here the type function checks the data type of the variable "message"

# Then the print function prints the result of the type function

print(message)

print(type(message)) # 'str' means string
print(5)

print(type(5)) # int means integer (whole numbers either positive or negative)



print(type(5.1))
print(True)

print(False)

print(1==1)
# Lists in Python are created by square brackets and can hold any value.

beatles = ["John", "Paul", "George", "Ringo"]

print(type(beatles))

print(beatles)
# .append on a list adds new values onto the end of the list.

beatles.append("Yoko")



# In Python notebooks, the last line of a cell can print a value automatically. (but only the last line)

beatles
# Exercise 1

# First, Create a new variable called "numbers" and assign it the numbers 1 through 9.

# Print your list of numbers.

numbers=[1,2,3,4,5,6,7,8,9]

numbers

# Exercise 2

# Add the number 10 onto your numbers variable. Be sure to use the Python method to add to the end of the list.

# Then print the numbers list

numbers.append(10)

numbers

# Exercise 3

# In this one cell, print out your new "numbers" variable, then the "beatles" variable, and also the "message" variable.a

print(numbers)

print(beatles)

print(message)
# Run this cell

import numpy as np    

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# This cell creates an empty dataframe

# Dataframes are like spreadsheets or database tables, and can hold TONS of data.

fruits = pd.DataFrame()
# Using square-brackets and a string, we can create or reference a column in the dataframe

# We can also assign lists and other data to each column.

fruits["name"] = ['apple', 'banana', 'crab apple', 'dewberry', 'elderberry', 'fig', 'guava', 'huckleberry', 'java apple', 'kiwi']

fruits
# Exercise 4

# Create a new column named "quantity" and assign it a list of 10 different quantities. 

# It's OK if there's 1 apple, 2 bananas, 3 crab apples, etc...

# print out your dataframe



fruits["quantity"]=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

fruits

# Let's assign a prices to each fruit

fruits["price"] = [1, .75, 0.10, 0.55, 0.65, 1.50, 2.00, 0.99, 1.99, 3.25]

# can write 0.75 to .75

fruits
# Let's do this together

# Delete the hashtag on the last line to uncomment it.

# Then run this cell. 



fruits["subtotal"] = fruits.price * fruits.quantity

fruits
# Let's print the dataframe to make sure we have:

# Subtotal, price, quantity, and fruit name.

print(fruits)
# Run this cell to create a new column where the entire tax column is 0.08

fruits["tax_rate"] = .08

fruits
# Uncomment the following line of code and run the cell to produce a tax_amount in dollars for each item

# Example

fruits["tax_amount"] = fruits.tax_rate * fruits.subtotal

fruits
# Exercise 5

# Create a new column named "total" then assign it the result of adding the "subtotal" and "tax_amount" column.

# Then print the dataframe

fruits["total"] = fruits.subtotal + fruits.tax_amount

fruits

# Let's check to see which of our fruits contains the string "berry"

fruits.name.str.contains("apple")
# If we use an array of booleans as a filter, we can "turn on" and "turn off" certain rows, and filter our results

fruits[fruits.name.str.contains("apple")]
# Exercise 6

# Use the syntax and operations introduced from the above example

# Show all of the rows that contain "berry" in the name.
df = sns.load_dataset("iris")

df
# Since we have width and length, let's try adding area as a "derived feature"

# Data scientists will often use the existing datatpoints to synthesize or derive new data that may add additional insight.

df["sepal_area"] = df.sepal_length * df.sepal_width

df["petal_area"] = df.petal_length * df.petal_width
# Let's visualize all of the measurement pairs and color by species

sns.pairplot(df, hue="species", corner=True)
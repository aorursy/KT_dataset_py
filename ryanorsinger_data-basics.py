# Variables point to values. 
# All variables have a value and a data type
# Strings can hold text, letters, and other characters
# Single = is the assigment operator 
message = "Howdy, Everybody!"
print(message) # The print function prints whatever you put in the parentheses
# Here the type function checks the data type of the variable "message"
# Then the print function prints the result of the type function
print(message)
print(type(message)) # 'str' means string
# There are different kinds of numbers
print(5)
print(type(5)) # int means integer (whole numbers either positive or negative)
print(type(5.0)) # float means a number with a "floating point" precision decimal decimal
# Comparison operators in Python, like == return True or False. Other math operators like < or > return True or False, too.
print(1 == 1)
print(type(True))
print(type(False))
print(True)
print(False)
# Lists in Python are created by square brackets and can hold any value.
beatles = ["John", "Paul", "George"]

print(type(beatles))
print(beatles)
# .append on a list adds new values onto the end of the list.
beatles.append("Ringo")

# In Python notebooks, the last line of a cell can print a value automatically. (but only the last line)
beatles
# Exercise 1
# First, Create a new variable called "numbers" and assign it the numbers 1 through 9 as a list
# Print your list of numbers.

# Exercise 2
# Add the number 10 onto your numbers variable. Be sure to use the Python method to add to the end of the list.
# Then print the numbers list

# Exercise 3
# In this one cell, print out your new "numbers" variable, then the "beatles" variable, and also the "message" variable, on their own line.

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
fruits["name"] = ['apple', 'banana', 'crab apple', 'dewberry', 'elderberry', 'fig', 'guava', 'huckleberry', 'java apple', 'kiwi', 'pineapple']
fruits
# Exercise 4
# Create a new column named "quantity" and assign it a list of 10 different quantities. 
# It's OK if there's 1 apple, 2 bananas, 3 crab apples, etc...
# print out your dataframe

# Let's assign some prices to each fruit
fruits["price"] = [1, .75, 0.10, 0.55, 0.65, 1.50, 2.00, 0.99, 1.99, 3.25, 1]
fruits
# Let's do this together
# Delete the hashtag on the last line to uncomment them. 
# Then run this cell. 
# fruits["subtotal"] = fruits.price * fruits.quantity
# Let's print the dataframe to make sure we have:
# Subtotal, price, quantity, and fruit name.
print(fruits)
# Run this cell to create a new column where the entire tax column is 0.08
fruits["tax_rate"] = .08
fruits
# Uncomment the last two lines of code and run this cell to produce a tax_amount in dollars for each item
# Example of creating a new column and setting it to be the result of multiplying two columns together
# fruits["tax_amount"] = fruits.tax_rate * fruits.subtotal
# fruits
# Exercise 5
# Create a new column named "total" then assign it the result of adding the "subtotal" and "tax_amount" column.
# Then print the dataframe

# Let's check to see which of our fruits contains the string "apple"
fruits.name.str.contains("apple")
# If we use an array of booleans as a filter, we can "turn on" and "turn off" certain rows, and filter our results
fruits[fruits.name.str.contains("apple")]
# Exercise 6
# Use the syntax and operations introduced from the above example
# Show all of the rows that contain "berry" in the name.

# Let's explore the data that we have some more
# .describe
# .max, .min
import seaborn as sns
df = sns.load_dataset("iris")
df
# Since we have width and length, let's try adding area as a "derived feature"
# Data scientists will often use the existing datatpoints to synthesize or derive new data that may add additional insight.
df["sepal_area"] = df.sepal_length * df.sepal_width
df
# Exercise 7 
# Create a new measurement called "petal_area" that contains the result of multiplying the petal_length by the petal_width values.

# Let's visualize all of the measurement pairs and color by species
sns.pairplot(df, hue="species", corner=True)

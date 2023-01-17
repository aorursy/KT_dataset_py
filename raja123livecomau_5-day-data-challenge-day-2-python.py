# Import our libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# read in our data
nutrition = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")

# look at only the numeric columns
nutrition.describe()
# This version will show all the columns, including non-numeric
# nutrition.describe(include="all")
# list all the coulmn names
print(nutrition.columns)

# get the sodium column
sodium = nutrition[" Sodium (mg)"]

# Plot a histogram of sodium content
plt.hist(sodium)
plt.title("Sodium in Starbucks Menu Items")
# list all the coulmn names
print(nutrition.columns)

# get the sodium column
fibre = nutrition[" Dietary Fibre (g)"]

# Plot a histogram of fibre content
plt.hist(fibre)
plt.title("Dietary Fibre in Starbucks Menu Items")
# Plot a histogram of sodium content with nine bins, a black edge 
# around the columns & at a larger size
plt.hist(sodium, bins=9, edgecolor = "black")
plt.title("Sodium in Starbucks Menu Items") # add a title
plt.xlabel("Sodium in milligrams") # label the x axes 
plt.ylabel("Count") # label the y axes
# Plot a histogram of fibre content with nine bins, a black edge 
# around the columns & at a larger size
plt.hist(fibre, bins=9, edgecolor = "black")
plt.title("Dietary Fibre in Starbucks Menu Items") # add a title
plt.xlabel("Fibre in grams") # label the x axes 
plt.ylabel("Count") # label the y axes
### another way of plotting a histogram (from the pandas plotting API)
# figsize is an argument to make it bigger
nutrition.hist(column= " Sodium (mg)", figsize = (12,12))
nutrition.hist(column= " Dietary Fibre (g)", figsize = (12,12))
nutrition.hist(column= " Total Carbohydrates (g) ", figsize = (12,12))
x = nutrition["Beverage_prep"] [:5]
y = nutrition["Caffeine (mg)"] [:5]
plt.xlabel("Beverage")
plt.ylabel("Caffeine (mg)")
plt.plot(x, y, marker="o")
plt.show()
# Import our libraries
import matplotlib.pyplot as plt
import pandas as pd

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
# Plot a histogram of sodium content with nine bins, a black edge 
# around the columns & at a larger size
plt.hist(sodium, bins=9, edgecolor = "black")
plt.title("Sodium in Starbucks Menu Items") # add a title
plt.xlabel("Sodium in milligrams") # label the x axes 
plt.ylabel("Count") # label the y axes
### another way of plotting a histogram (from the pandas plotting API)
# figsize is an argument to make it bigger
nutrition.hist(column= " Sodium (mg)", figsize = (12,12))
# import our libraries

import matplotlib.pyplot as plt

import pandas as pd



# read our data

nutrition = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")



#look at only the numeric column

nutrition.describe()



# this version will show all the columns including non numeric

nutrition.describe(include = "all")

# list all the columns names 

# nutrition.columns

# nutrition["Calories"]

# plt.hist(nutrition["Calories"])



nutrition[" Sodium (mg)"]

# plot a histogram  of sodium content

plt.hist(nutrition[" Sodium (mg)"])

plt.title("Sodium in Starbucks Menu Items")
# another way of plotting a histogram (from the pandas API)

nutrition.hist(column =" Sodium (mg)")
sodium = nutrition[" Sodium (mg)"]

plt.hist(sodium, bins = 9, edgecolor = "black")

plt.xlabel("Sodium in milligrams")

plt.ylabel("Count")
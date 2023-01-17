#import numpy as np # linear algebra

import pandas as pd     # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# read in our data

nutrition = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")



# Look at only the numeric column

nutrition.describe()

# This version will show all the columns, including non-numeric

# nutrition.describe(include="all") # SHOW ALL COLUMNS
# list all the column names

print(nutrition.columns) 



# get the sodium column

sodium = nutrition[" Sodium (mg)"]





# Plot a histogram of sodium content

plt.hist(sodium)



#plt.hist(nutrition[" Sodium (mg)"]) 

plt.title("Sodium in Starbucks Menu Item")
# Plot a histogram of sodium content with bins, a black edge around the columns 

# & at larger size  

plt.hist(sodium, bins=9, edgecolor = "black")



#plt.hist(nutrition[" Sodium (mg)"]) 

plt.title("Sodium in Starbucks Menu Item")  # add a title

plt.xlabel("Sodium in milligrams") # label the x axes

plt.ylabel("Count") # label the y axes
# another way of plotting a histogram (from the pandas plotting API)

# figsize is an argument to make it bigger

nutrition.hist(column= " Sodium (mg)", figsize= (12,12))
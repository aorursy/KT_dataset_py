# import the libraries

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")

# look at the numeric data

data.describe().transpose()

# data.describe(include="all").transpose() - will show all the columns including non-numeric
plt.hist(data[" Sodium (mg)"])

plt.title("Sodium in Starbucks menu items")
# plot with 9 bins 

# black edge around the bins

plt.hist(data[" Sodium (mg)"], bins = 9, edgecolor = "black") 

plt.title("Sodium in Starbucks menu items") # add a title

plt.xlabel("Sodium in miligrams")

plt.ylabel("Count")
# another way of plotting a histogram from pandas api

data.hist(column = " Sodium (mg)", figsize = (12,12))
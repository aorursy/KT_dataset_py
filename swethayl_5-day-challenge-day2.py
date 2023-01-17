# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# read in our data
cereals = pd.read_csv("../input/cereal.csv")
# look at only the numeric columns
cereals.describe()
# This version will show all the columns, including non-numeric
# list all the coulmn names
print(cereals.columns)
# get the calories column
calories = cereals["calories"]
# Plot a histogram of calories content
plt.hist(calories)
plt.title("Calories in Cereals")
# Plot a histogram of calories content with nine bins, a black edge 
# around the columns & at a larger size
plt.hist(calories, bins=9, edgecolor = "yellow")
plt.title("Calories in Cereals") # add a title
plt.xlabel("Calories per Serving") # label the x axes 
plt.ylabel("Number of Cereals") # label the y axes
# get the calories column
sodium = cereals["sodium"]
# Plot a histogram of calories content
#plt.hist(sodium)
plt.title("Sodium content in Cereals")
plt.hist(sodium, bins=9, edgecolor = "red")
plt.xlabel("Sodium content (mg)") # label the x axes 
plt.ylabel("Number of Cereals") # label the y axes
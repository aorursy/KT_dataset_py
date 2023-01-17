# import libraries
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read in the data set
cereal = pd.read_csv("../input/cereal.csv")

# list all the column names
cereal.columns
# get the sodium column
sodium = cereal["sodium"]

# check for missing values
print("Missing values = {}".format(sodium.isnull().sum()))

# summary of the sodium column
sodium.describe()

# Plot a histogram of sodium content with nine bins
# with black edge around the columns & at a larger size

plt.figure(figsize = (12,12)) # set size at 12 inches x 12 inches
plt.hist(sodium, bins=9, edgecolor = "black") # set nine bins and black edge
plt.title("Sodium in 80 cereal products") # add a title
plt.xlabel("Sodium in milligrams") # label the x axes 
plt.ylabel("Count") # label the y axes
# another way of plotting a histogram (from the pandas plotting API)
# figsize is an argument to make it bigger
cereal.hist(column= "sodium", figsize = (12,12), bins=9, edgecolor = "black")
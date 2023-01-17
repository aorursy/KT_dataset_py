# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # Importing NumPy library
import pandas as pd # Importing Pandas library
import matplotlib.pyplot as plt # Importing Matplotlib library's "pyplot" module
import seaborn as sns # Imorting Seaborn library

# Input data files are available in the "../input/" directory.

# This lines for Kaggle:
import os
print(os.listdir("../input"))
data = pd.read_csv("../input/bookfeatures.csv") # Read CSV file and load into "data" variable
data.info() # Show detailed information for dataset columns(attributes)
data.corr() # This method shows the correlation matrix of the columns (attributes)
fig, axes = plt.subplots(figsize=(18, 18)) # This method creates a figure and a set of subplots. Then, returns figure and axis infos.
sns.heatmap(data=data.corr(), annot=True, linewidths=.5, fmt=".1f", ax=axes) # Figure out heatmap
# Parameters:
# data : 2D data for the heatmap.
# annot : If True, write the data value in each cell.
# linewidths : Width of the lines that will divide each cell.
# fmt : String formatting code to use when adding annotations.
# ax : Axes in which to draw the plot, otherwise use the currently-active Axes.
plt.show() # Shows only plot and remove other info
data.head(10) # Returns first 10 entries
data.columns # Returns column names as a list
# Line Plot
# Plot all Average Rating values of the Dataset
data.avgRating.plot(kind='line', color='r', label='Average Rating', linewidth=1, alpha=0.7, grid=True, linestyle = '-', figsize=(18,9))
plt.legend(loc='upper right')     # Location of the legend
plt.xlabel('X Axis')              # Label of the X axis
plt.ylabel('Y Axis')              # Label of the Y axis
plt.title('Ratings')            # Title of the overall plot
plt.show()
# Scatter Plot
# x = rating1, y = rating_count
data.plot(kind='scatter', x='rating1', y='rating_count', alpha=0.8, color='green', figsize=(18,9))
plt.xlabel('Rating 1 Star')              # Label of the X axis
plt.ylabel('Rating Count')               # Label of the y axis
plt.title('Rating 1 Star to All Rating Scatter Plot') # Title of the all plot
plt.show()
# Histogram
# Frequency of all average ratings
# bins : Number of bars in figure
data.avgRating.plot(kind='hist', bins=50, figsize = (18,9))
plt.show()
dict = {'germany' : 'berlin', 'usa' : 'washington'} # Create a dictionary
print(dict.keys()) # Print its keys
print(dict.values()) # Print its values
dict["germany"] = "munich" # Updating existing entry
print(dict)
dict["france"] = "paris" # Adding new entry
print(dict)
del dict["usa"] # Removing existing entry
print(dict)
print("france" in dict) # Check dictionary key existence
dict.clear() # Remove all elements of dictionary
print(dict)
del dict # Remove dictionary itself
data.columns
series = data['avgRating'] # Data which place in "avgRating" column return as a Series data type 
print(type(series)) # Prints type of "series" variable
data_frame = data[['avgRating']] # Data which place in "avgRating" column return as a DataFrame data type
print(type(data_frame)) # Prints type of "data_frame" variable
x = data['avgRating'] > 4.0 # Filtering DataFrame and return filtered data
data[x].head(10) # Print first 10 filtered data
# Get data which provides condition "avgRating > 4.2" AND "rating_count > 20000"
data[np.logical_and(data['avgRating'] > 4.2, data['rating_count'] > 20000)]
# Get data which provides condition "avgRating > 4.2" AND "rating_count > 20000"
data[(data['avgRating'] > 4.2) & (data['rating_count'] > 20000)]
# Print elements of the list:
lis = [1, 2, 3, 4, 5, 6]
for i in lis:
    print("i is: ", i)
print('')

# Enumerating list elements and print they:
for index, value in enumerate(lis):
    print(index, " : ", value)
print('')

# Print dictionary elements:
dict = {'spain' : 'madrid', 'usa' : 'vegas'}
for key, value in dict.items():
    print(key, " : ", value)
print('')

# Iterating on DataFrame:
for index, value in data[['avgRating']][0:1].iterrows():
    print(index, " : ", value)
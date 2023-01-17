# pull in libraries and data source

# pull in anscombe's quartet and visualize
# These libraries are the bread and butter Python Data Science libraries

import numpy as np               # numpy is for fast, numerical calculation and 

import pandas as pd              # pandas is like Excel on steroids for statistical analysis

import matplotlib.pyplot as plt  # matplotlib is the 

import seaborn as sns            # seaborn integrates with matplotlib and pandas

from pydataset import data       # pydataset is a few hundred data sets for practicing data science methods

%matplotlib inline               
quartet = sns.load_dataset("anscombe") # Anscombe's quartet is a data that hightights the value of visualizations
quartet.head() # There are 4 sets of data in this data set, each with their own X and Y values
quartet.tail()
first = quartet[quartet["dataset"] == "I"]

second = quartet[quartet["dataset"] == "II"]

third = quartet[quartet["dataset"] == "III"]

fourth = quartet[quartet["dataset"] == "IV"]
## Descriptive Statistics

print("First set:")

print(first.describe())

print("-------")

print()



print("Second set:")

print(second.describe())

print("-------")

print()



print("Third set:")

print(third.describe())

print("-------")

print()



print("Fourth set:")

print(fourth.describe())
# Let's find out! Run this cell to graph the X and Y values from each quartet

sns.relplot(x='x', y='y', col='dataset', data=quartet)
# walk through mpg.csv together for some aggregate stuff

# do the exercises I've already built here, yay!



# pull in lemonade.csv

# Together, create the derived values for revenue and add to frame df['revenue'] = ...

# Together, create the month name from the numeric date values df['month'] = ...



## Exercises

# what's the highest day for revenue?

# which month had the best revenue?
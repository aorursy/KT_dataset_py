## Install packages needed for this Capstone!
import pandas as pd #hey, look... pandas! :) You know this package!
import matplotlib #you have not seen this yet, but that's OK! it's a graphing package :)
import seaborn as sns #another graphing package... can be prettier than matplotlib!
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# This is helpful for you to know if you have added your data, but we did that for you!
import os
print(os.listdir("../input"))

# Results listed should be the 3 datasets from https://openpolicing.stanford.edu/data
# Colorado, Vermont, Massachusetts
#read in the VERMONT data set below and print the head to make sure you get the data!
#format the date column

#format the time column

#create new column that merges together date and time to create a time stamp

#extract the day of the week and hour of the day

#take a second to note the *new* last 3 columns that our code before this made!
# use .head()

#find the unique variables existing in a column

#find the variable counts of the unique values in the code above

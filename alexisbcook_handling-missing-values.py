# modules we'll use

import pandas as pd

import numpy as np



# read in all our data

nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")



# set seed for reproducibility

np.random.seed(0) 
# look at the first five rows of the nfl_data file. 

# I can see a handful of missing data already!

nfl_data.head()
# get the number of missing data points per column

missing_values_count = nfl_data.isnull().sum()



# look at the # of missing points in the first ten columns

missing_values_count[0:10]
# how many total missing values do we have?

total_cells = np.product(nfl_data.shape)

total_missing = missing_values_count.sum()



# percent of data that is missing

percent_missing = (total_missing/total_cells) * 100

print(percent_missing)
# look at the # of missing points in the first ten columns

missing_values_count[0:10]
# remove all the rows that contain a missing value

nfl_data.dropna()
# remove all columns with at least one missing value

columns_with_na_dropped = nfl_data.dropna(axis=1)

columns_with_na_dropped.head()
# just how much data did we lose?

print("Columns in original dataset: %d \n" % nfl_data.shape[1])

print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# get a small subset of the NFL dataset

subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()

subset_nfl_data
# replace all NA's with 0

subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 

# then replace all the remaining na's with 0

subset_nfl_data.fillna(method='bfill', axis=0).fillna(0)
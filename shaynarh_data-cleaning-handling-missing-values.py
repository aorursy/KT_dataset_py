import pandas as pd
import numpy as np
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
sf_permits.sample(5)
# get the number of missing data points per column
missing_values_count = sf_permits.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(sf_permits.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# option 1: remove all columns with at least one missing value
columns_with_na_dropped = sf_permits.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
#or you can replace values

# replace all NA's with 0
sf_permits.fillna(0).head(5)
# or can replace all NA's the value that comes directly after it in the same column,
#makes more sense in dfs that have some sort of logical order
# then replace all the reamining na's with 0
sf_permits.fillna(method = 'bfill', axis=0).fillna(0)
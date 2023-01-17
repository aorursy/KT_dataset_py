# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
nfl_data.sample(5)
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?
# your code goes here :)
sf_permits.sample(6)
# get the number of missing data points per column
nfldata_missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
nfldata_missing_values_count[0:10]
# how many total missing values do we have?
nfldata_total_cells = np.product(nfl_data.shape)
nfldata_total_missing = nfldata_missing_values_count.sum()

# percent of data that is missing
(nfldata_total_missing/nfldata_total_cells) * 100
# your turn! Find out what percent of the sf_permit dataset is missing

sf_permits_missing_values_count = sf_permits.isnull().sum()
#print(sf_permits_missing_values_count)
sf_permits_total_cells = np.product(sf_permits.shape)
#print(sf_permits_total_cells)
sf_permits_total_missing = sf_permits_missing_values_count.sum()
#print(sf_permits_total_missing)

# percent of data that is missing
(sf_permits_total_missing/sf_permits_total_cells) * 100
# look at the # of missing points in the first ten columns
nfldata_missing_values_count[0:10]
# `Street Number Suffix`-- Related to address.
# Missing values for this column may not exist, because the Street Number may not have suffixes

# `Zipcode`-- Zipcode of building address.
# The missing value of this column should be unrecorded, because the zip code of the address must have

print(sf_permits_missing_values_count[6:8])
print('-' * 32)
print(sf_permits_missing_values_count[38:41])
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
nfldata_columns_with_na_dropped = nfl_data.dropna(axis=1)
nfldata_columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset(nfl_data): %d \n" % nfl_data.shape[1])
print("Columns with na's dropped(nfl_data): %d" % nfldata_columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits.dropna()
# Now try removing all the columns with empty values. Now how much of your data is left?
sf_permits_columns_with_na_dropped = sf_permits.dropna(axis=1)
sf_permits_columns_with_na_dropped.head()
print("Columns in original dataset(sf_permits): %d \n" % sf_permits.shape[1])
print("Columns with na's dropped(sf_permits): %d" % sf_permits_columns_with_na_dropped.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
# Your turn! Try replacing all the NaN's in the sf_permit data with the one that
# comes directly after it and then 

sf_permits_copy1 = sf_permits.copy()
sf_permits_copy1[0:10]
sf_permits_afterit = sf_permits_copy1.fillna(method = 'ffill', axis=0, limit = 2).fillna("0")
sf_permits_afterit[0:10]
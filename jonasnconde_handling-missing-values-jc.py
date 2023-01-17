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
sf_permits.sample(2)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permit dataset is missing

missing_values_count_sf = sf_permits.isnull().sum()

total_cells_sf = np.product(sf_permits.shape)
total_missing_sf = missing_values_count_sf.sum()

# percent of data that is missing
(total_missing_sf/total_cells_sf) * 100
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
print(sf_permits["Zipcode"].isnull().sum())

print(sf_permits["Street Number Suffix"].isnull().sum())

print(len(sf_permits))

# Zipcode weren't recorded because almost every address contains a Zipcode.
# Street Number Suffix don't exists because not every address contains a suffix in their street number, as we can see that the number of NaN values
# are very close to the total lenght of the dataset.
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits.dropna()

# All rows were removed!
# Now try removing all the columns with empty values. Now how much of your data is left?
columns_with_na_dropped_sf = sf_permits.dropna(axis=1)
columns_with_na_dropped_sf.head()

print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped_sf.shape[1])

# We lost a lot of columns!
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

sf_permits.fillna(method = 'bfill', axis=0).fillna("0")
# I added the OpenAddresses dataset to the my kernel by clicking on [>], "Add Data Source",
# and choosing the OpenAddresses - U.S. West.

# Goal: find the Zipcodes for the missing values in sf_permits.
#UPDATE: There is no San Francisco data into the "ca.csv".

# First, let's create a sf_permits_subset with only the rows with NaN values in Zipcode column.
sf_permits_subset = sf_permits.loc[sf_permits["Zipcode"].isnull() == True]
sf_permits_subset.head()
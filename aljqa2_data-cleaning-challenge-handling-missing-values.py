# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
#nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 
sf_permits.head()
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
nfl_data.sample(5)
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?
sf_permits.sample(5)
# your code goes here :)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
#len(missing_values_count)
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()
#total_cells
# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing
missing_values_counts_sf = sf_permits.isnull().sum()
#missing_values_counts_sf[:10]
total_cells = np.product(sf_permits.shape)
total_missing = missing_values_counts_sf.sum()
(total_missing/total_cells) * 100
missing_values_counts_sf
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# remove all the rows that contain a missing value
#nfl_data.dropna()
#print(len(nfl_data))
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
#print(len(columns_with_na_dropped))
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
rows_with_na_dropped = sf_permits.dropna(axis=0,how = 'any')
#rows_with_na_dropped.head()
print("Number of rows in original dataset: %d \n" % sf_permits.shape[0])
print("Number of rows with na's dropped: %d \n" % rows_with_na_dropped.shape[0])
# Now try removing all the columns with empty values. Now how much of your data is left?
columns_with_na_dropped = sf_permits.dropna(axis=1)
print("Number of columns in original dataset: %d \n" % sf_permits.shape[1])
print("Number of columns with na's dropped: %d \n" % columns_with_na_dropped.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing any remaining NaN's with 0
sf_permits.fillna(method = 'bfill', axis=0).fillna(0).head()
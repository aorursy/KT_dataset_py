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
sf_permits.sample(5)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
#(nfl_data.shape) gives the number of rows (n) and columns (m)
#np.product(nfl_data.shapes) tells n * m = number of cells in the array
total_missing = missing_values_count.sum()
# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing
missing_sfvalues_count = sf_permits.isnull().sum()
# total number of missing values per column
total_sfcells = np.product(sf_permits.shape)
#product of the number rows times the number of columns = the number of cells in the array
total_sfmissing = missing_sfvalues_count.sum()
#total number of missing values
percentage_sfmissing= (total_sfmissing/total_sfcells)*100
print(percentage_sfmissing)

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
# axis=1 represents the columns where axis=0 would have represented the rows and is the default setting
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# using [0] instead of [1] gives the number of rows and this should and is equal in both sets
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits.dropna()
# Now try removing all the columns with empty values. Now how much of your data is left?
sfcolumns_with_na_dropped = sf_permits.dropna(axis=1)
sfcolumns_with_na_dropped.head()

print("Number of columns in original dataset: %d \n" % sf_permits.shape[1])
print("Number of columns with na's dropped: %d \n" % sfcolumns_with_na_dropped.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
#bfill = backwards filling, there is also forward filling (ffill)
# using axis=0 because it takes the value from the next ROW in the same column
# when axis=1 the value from the next COLUMN is taken to fill in the NA value
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then 
sf_permits_subset=sf_permits.head()
sf_permits_subset.fillna(method= 'bfill', axis=0).fillna(0)
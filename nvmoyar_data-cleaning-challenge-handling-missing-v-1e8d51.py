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
sf_permits.head()
sf_permits.info() # results: 43 columns
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# Create a pandas series with the count of the sf_permits dataset is missing per column
sf_permits_missing_values_count = sf_permits.isnull().sum() 

# The size of this pandas series is 43 since we have 43 columns, but not all the columns have missing values. 
# Get the first 10 elements of the pandas series with null entries
print(sf_permits_missing_values_count[:10]) 
# Get the total number of elements missing
sf_permit_total_missing = sf_permits_missing_values_count.sum()

# Get only the column with null entries
print("\nCol's with missing data\n\n", sf_permits_missing_values_count[sf_permits_missing_values_count != 0])
print("\nCol's with 90% missing data\n\n", sf_permits_missing_values_count[sf_permits_missing_values_count > 179010]) 
print("\nTotal col's with missing data: ", sf_permits_missing_values_count[sf_permits_missing_values_count != 0].size)
print("\nTotal missing data: ", sf_permit_total_missing)
# how many total missing values do we have?
sf_permit_total_cells = np.product(sf_permits.shape) # matrix of (198900, 43) 

# percent of data that is missing respect to the whole dataset
(sf_permit_total_missing/sf_permit_total_cells) * 100
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# remove all the rows that contain a missing value
nfl_data.dropna()
nfl_data.head()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits.dropna()
sf_permits.head()
# Now try removing all the columns with empty values. Now how much of your data is left?

sf_permits_columns_with_na_dropped = sf_permits.dropna(axis=1)
sf_permits_columns_with_na_dropped.head()
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# Try replacing all the NaN's in the sf_permits data with the one that 
# comes directly after it and then replacing any remaining NaN's with 0

sf_permits.fillna(method = 'bfill', axis=0).fillna(0) # axis=0 means vertically, so fill a missing value with the next under it

# that makes sense in columns that show any kind of logic correlation. This method adds more redundancy in data, but at least we can
# compute some calculations instead of discarding this missing values. Another approach could be filling with the mean, etc. when that
# makes sense. There is no magic formula here, observation and instituition. 

sf_permits.fillna(0) # this way we remove NaN and we have same type through all the same column

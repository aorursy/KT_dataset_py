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
# This is my first ever line of Pyhton code coming from an R user.
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
total_cells_permits = np.product(sf_permits.shape)
missing_values_count_permits = sf_permits.isnull().sum()
total_missing_permits = missing_values_count_permits.sum()

(total_missing_permits/total_cells_permits)*100

# This was okay because the code demonstrated for the nfl_data is transferable. I like Python's syntax for calling functions to dataframes.
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
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
total_rows_permits = len(sf_permits)
total_rows_permits
rows_with_na_dropped_permits = len(sf_permits.dropna())
rows_with_na_dropped_permits
print("Rows in original dataset: %d \n" % total_rows_permits)
print("Rows with na's dropped: %d" % rows_with_na_dropped_permits)

# This took a bit longer as I wanted to copy the format of the previous code example to show the original number of rows in the dataframe 
# and also those missing.
# After consulting stack overflow I found there is a simple solution that is computationally the best in this scenario (not my words) - len()
# This counts the number of rows in the dataframe.

# Now try removing all the columns with empty values. Now how much of your data is left?
columns_with_na_dropped = sf_permits.dropna(axis=1)
columns_with_na_dropped.head()
print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])

# More tranferable code from the previous cell and just swapping the name to sf_permits
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
sf_permits.sample(5)
subset_permits_data = sf_permits.loc[:, 'Existing Construction Type':'Record ID'].head(100)
subset_permits_data
subset_permits_data.fillna(method = 'bfill', axis=0)

# After once more consulting Stackoverflow I deduced that 'bfill' was a proxy for Backwards Fill? Maybe I'm wrong but it seems to have worked.
# There is also 'ffill' which I belive to be forward fill
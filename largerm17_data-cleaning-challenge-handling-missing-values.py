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
sf_permits.sample(7)
# there is missing data in sf_permits also !
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
sf_permits_missing_values_count = sf_permits.isnull().sum()
# check number of missing values in first 15 columns
sf_permits_missing_values_count[:15]
# get total number of data records also
len(sf_permits)
# finding percentage of data missing
missing_cells = sf_permits_missing_values_count.sum()
total_cells = np.product(sf_permits.shape)
(missing_cells / total_cells) * 100
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
sf_permits_missing_values_count['Street Number Suffix']
# from the data, it seems that 'Street Number Suffix' column has missing values because they don't exist
sf_permits_missing_values_count['Zipcode']
# in 'Zipcode' column, some values are missing because they were not recorded 
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
# Each row has at least one missing value
# Now try removing all the columns with empty values. Now how much of your data is left?
complete_columns = sf_permits.dropna(axis=1)
complete_columns.head()
print('Columns in original sf dataset: %d' % sf_permits.shape[1])
print('Columns after removing na: %d' % complete_columns.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
sf_permits.head()
# Your turn! Try replacing all the NaN's in the sf_permit data with the one that
# comes directly after it and then 
sf_permits.fillna(method='bfill', axis=0).fillna('0').head()
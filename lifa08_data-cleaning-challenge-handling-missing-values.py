# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
nfl_data = pd.read_csv('../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv')
sf_permits = pd.read_csv('../input/building-permit-applications-data/Building_Permits.csv')

# set seed for reproducibility
np.random.seed(0)
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
nfl_data.sample(5)
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?
sf_permits.sample(6)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing
print(sf_permits.shape)
total_cells = np.product(sf_permits.shape)

missing_values_cnt_sf_permits = sf_permits.isnull().sum()
total_missing = missing_values_cnt_sf_permits.sum()
print(missing_values_cnt_sf_permits.size)
print(missing_values_cnt_sf_permits[:9])

missing_percent = total_missing/total_cells
print(missing_percent)
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# It is highly possible that Street Number Suffix don't exist,
# while the Zipcode are missing because they weren't recorded.
print(missing_values_cnt_sf_permits['Street Number Suffix'])
print(missing_values_cnt_sf_permits['Zipcode'])
print(sf_permits.shape[0])
# print(missing_values_cnt_sf_permits.index)
# print(sf_permits['Zipcode'])
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
# No even one row has been left.
# Now try removing all the columns with empty values. Now how much of your data is left?
sf_column_drop = sf_permits.dropna(axis=1)
print('The number of columns of original dataset:  %d\n' %sf_permits.shape[1])
print('The number of columns of dataset that has drop columns: %d \n' %sf_column_drop.shape[1])

# More than half number of columns has been drop!
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

# Delete columns with lots of missing values
cols_with_huge_missing = [col for col in sf_permits.columns if sf_permits[col].isnull().sum() > 100000]
sf_permits_reduced = sf_permits.drop(cols_with_huge_missing, axis=1)

cols_with_missing = sf_permits_reduced.isnull().sum().index
# columns that has numeric values
cols_with_missing_numeric = [col for col in cols_with_missing 
                             if sf_permits_reduced[col].dtype == np.int64 
                             or sf_permits_reduced[col].dtype == np.float64 ]

# columns that has non-numeric values
cols_with_missing_non_numeric = [col for col in cols_with_missing.values 
                                 if col not in cols_with_missing_numeric]

# fills numeric missing values with the column mean
for col in cols_with_missing_numeric:
    col_mean = np.mean(sf_permits_reduced[col])
    sf_permits_reduced[col] = sf_permits_reduced[col].fillna(col_mean)
print(sf_permits_reduced[cols_with_missing_numeric].isnull().sum())

# print(sf_permits_imputed_numeric)
for col in cols_with_missing_non_numeric:
    sf_permits_reduced[col] = sf_permits_reduced[col].fillna(method = 'bfill', axis=0).fillna(0)
print(sf_permits_reduced[cols_with_missing_non_numeric].isnull().sum())

sf_permits_reduced.sample(10)
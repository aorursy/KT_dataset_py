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
# here we are looking at 10 rows of the sf permits data set. We can already see that the Site Permit column has a number of NaN values.
sf_permits.sample(10)
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
missing_val_count = sf_permits.isnull().sum()
missing_val_count
# Here, it looks like we are missing a large number of fields in specific columns
# lets first look at the percentage missing values in each column before looking at the total percentage of missing values:
total_rows = len(sf_permits.index)
missing_col_per = (missing_val_count /total_rows) * 100
missing_col_per
#now we are going to look at the total missing percentage
total_cells = np.product(sf_permits.shape)
total_missing = missing_val_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
#Looking at the rows where the street number suffix is not null.
numbersuffix_notnull = sf_permits.loc[sf_permits['Street Number Suffix'].notnull(), ['Street Number','Street Number Suffix','Street Name','Street Suffix','Description','Zipcode']]
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
null_row_removed = sf_permits.dropna()
print("Rows in original dataset: %d \n" % total_rows)
print("Rows remaining: %d \n" % len(null_row_removed.index))
# Now try removing all the columns with empty values. Now how much of your data is left?
null_cols_removed = sf_permits.dropna(axis=1)
null_cols_removed.head()
#And exactly how many columns did we get rid of?
print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % null_cols_removed.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then 
filled_in = sf_permits.fillna(method = 'bfill', axis=0).fillna("0")
filled_in.head()
#and as you can guess, this will make some columns like `Street Number Suffix` that 
#we looked at get filled with a large number of values where it might make sense to leave those values as NaN for them not existing.
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
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permit dataset is missing
missing_values_per_column = sf_permits.isnull().sum()
missing_values_per_column[0:10]
total_missing_sf_permits = missing_values_per_column.sum()
# percent_missing_sf_permits:
(total_missing_sf_permits/np.product(sf_permits.shape))*100
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# Out of total 198900 values, 196684 are Nan
sf_permits.xs(key="Street Number Suffix", axis=1).shape
sf_permits.xs(key="Street Number Suffix", axis=1).isnull().sum()
# Print Non NaN values
non_null_rows_list = sf_permits.index[np.where(sf_permits.xs(key="Street Number Suffix", axis=1).notnull())[0]]
func_ssn = lambda x: x=="Street Number Suffix"
street_suffix_column_number = np.where(pd.Series(sf_permits.columns).apply(func_ssn).tolist())[0]
sf_permits.iloc[non_null_rows_list, street_suffix_column_number]
# ZipCodes
# Get the column for Zipcode
func_zip = lambda x: x=="Zipcode"
zipcode_column_number = np.where(pd.Series(sf_permits.columns).apply(func_zip).tolist())[0]
# Count of null values in Zipcode column; 1716/198900
sf_permits.xs(key="Zipcode", axis=1).isnull().sum()
# Print non nul zips
non_null_rows_list_zip = sf_permits.index[np.where(sf_permits.xs(key="Zipcode", axis=1).notnull())[0]]
sf_permits.iloc[non_null_rows_list_zip, zipcode_column_number]
# sf_permits.loc[non_null_rows_list_zip, "Zipcode"]

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
# 0 rows
# Now try removing all the columns with empty values. Now how much of your data is left?
columns_with_na_dropped_sf_permits = sf_permits.dropna(axis=1)
columns_with_na_dropped_sf_permits.sample(5)
print("Originally sf_permits shape: ", sf_permits.shape)
print("After dropping NA colums, sf_permits shape: ", columns_with_na_dropped_sf_permits.shape)
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
sf_permits.fillna(method='bfill', axis=0).fillna("0")
# Doesn't make sense to fill all the NaNs with 0, but still
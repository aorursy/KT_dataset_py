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
# Seems that sf_permits has some missing values
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
missing_values_count_sf_permits = sf_permits.isnull().sum()

total_cells = np.product(sf_permits.shape)
total_missing = missing_values_count_sf_permits.sum()

(total_missing/total_cells) * 100
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
#Checking the number of nulls for each column
total_null_strt_numb_suffix = sf_permits[['Street Number Suffix']].isnull().sum()
total_null_zipcode = sf_permits[['Zipcode']].isnull().sum()

print(total_null_strt_numb_suffix)
print(total_null_zipcode)
print("percentage of missing",(total_null_strt_numb_suffix/np.product(sf_permits.shape[0])) * 100)
print("percentage ofmissing",(total_null_zipcode/np.product(sf_permits.shape[0])) * 100)

#It is obvious that the percentage related to the Street number suffix is very high and the percentage of the zipcode is too low 
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
# Now try removing all the columns with empty values. Now how much of your data is left?
column_with_na_dropped_sf_permit = sf_permits.dropna(axis=1)

#Just to make a small comparison
print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns in original dataset: %d \n" % column_with_na_dropped_sf_permit.shape[1])
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
sf_permits.fillna(method = 'bfill', axis=0).fillna("0").head()
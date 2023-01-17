# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
# Rachael will be using NFL data
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
# We will be using SF Building Permits data
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
nfl_data.sample(5)
# the same for SF data
sf_permits.sample(7)
# Basic info table: N of rows, N of columns
sf_permits.shape

# Some alternative way to look into data
sf_permits.describe()
# This provide statistics information about the sample 
# Note that 'count' (i.e. the number of valid entries) is different for each column (aka 'feature'), 
# this is beacause of different number of NaN/None values within the data.

# another alternative
# show first N rows:
sf_permits.head(5)
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?
sf_permits.sample(3)

# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count.sum
# get the number of missing data points per column
# missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permit dataset is missing

# Step 1

# get the number of missing data points per column
missing_values_count_sf = sf_permits.isnull().sum()

# look at the # of missing points in the first ten columns
# missing_values_count 
missing_values_count_sf[0:10]
# Step 2

# how many total missing values do we have?
total_cells_sf = np.product(sf_permits.shape)
total_missing_sf = missing_values_count_sf.sum()

# percent of data that is missing
missp=(total_missing_sf/total_cells_sf) * 100

# add some formatting to the results
print('Percentage of missing values in the Data (sf_pemits): '+str(round(missp,1))+' %')
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# take another look at the data
sf_permits.describe()
# actual data sampling
sf_permits.sample(5)
sf_permits.shape
# look at the # of missing points in the first ten columns
missing_values_count_sf[0:10]
    
v1=missing_values_count_sf['Zipcode']
v2=missing_values_count_sf['Street Number Suffix']

print(v1,v2)
# remove all the rows that contain a missing value
nfl_data.dropna()
# The same for sf_permits data
sf_permits.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits.dropna() # equivalent to : sf_permits.dropna(axis=0)

# Now try removing all the columns with empty values. Now how much of your data is left?
# remove all columns with at least one missing value
columns_with_na_dropped_sf = sf_permits.dropna(axis=1)
columns_with_na_dropped_sf.head()
# just how much data did we lose in sf_permits data?
print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped_sf.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head(10)
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
# Your turn! Try replacing all the NaN's in the sf_permit data with the one that
# comes directly after it and then (zeroes?)
sf_permits.fillna("0").head() # just substitute 0 to NaN
sf_permits.fillna(method = 'bfill', axis=0).fillna("0").head() # imputation by assuming the next value in the same column
# sf_permits.fillna(method = 'bfill', axis=1).fillna("0").head() 
# NOTE: above we try with 'axis=1', this makes to substitute with the value in the next column

# you can work also on individual columns
# sf_permits[["Street Number Suffix","Site Permit"]].fillna(method = 'bfill', axis=0).fillna("0").head() # imputation by assuming the next value in the same column

# and compare to the origina table
sf_permits.head()
#sf_permits.loc[sf_permits.zipcode.isnull(), 'Zipcode'] = sf_permits.loc[sf_permits.zipcode.isnull(), 'id'].map(sf_permits.loc[sf_permits.paid_date.notnull()].set_index('Street Name')['Zipcode'])
new=sf_permits['Zipcode'].groupby(by=sf_permits['Street Name']).ffill()
#sf_permits[['Zipcode','Street Name']].head(10)
sf_permits['Zipcode'].isnull().sum()
new.isnull().sum() # still 34 nan values (maybe unique street names?), can be improved using different columns for grouping
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
sf_permits.sample(6)
# your code goes here :)
# get the number of missing data points per column
#missing_values_count = nfl_data.isnull().sum()
missing_values_count = nfl_data.isnull().sum()
# look at the # of missing points in the first ten columns
missing_values_count[0:20]
#missing_values_count.head()
#type(missing_values_count)
nfl_data.shape
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()
print(total_cells)
# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing
print(sf_permits.shape)
sf_permits_count=sf_permits.isnull().sum()
print(sf_permits_count[0:10])

sf_permits_all_cell=np.product(sf_permits.shape)
print(sf_permits_all_cell)
sf_missing=sf_permits_count.sum()
print(sf_missing)
sf_missing_rate=sf_missing/sf_permits_all_cell*100
print(sf_missing_rate)
# look at the # of missing points in the first ten columns
missing_values_count[0:10]

# remove all the rows that contain a missing value
nfl_no_na=nfl_data.dropna()
nfl_no_na
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
print(nfl_data.shape)
columns_with_na_dropped.head()
#print(columns_with_na_dropped.shape)
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits_drop_cloume=sf_permits.dropna(axis=0)
sf_permits_drop_cloume.head()
sf_permits.shape[1]
print("row in original dataset: %d \n" % sf_permits.shape[0])
sf_permits_drop_cloume.shape[1]
print("row with na's dropped: %d" % sf_permits_drop_cloume.shape[0])
# Now try removing all the columns with empty values. Now how much of your data is left?
sf_permits_drop_cloume=sf_permits.dropna(axis=1)
sf_permits_drop_cloume.head()
sf_permits.shape[1]
print("colume in original dataset: %d \n" % sf_permits.shape[1])
sf_permits_drop_cloume.shape[1]
print("colume with na's dropped: %d" % sf_permits_drop_cloume.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
sf_sub=sf_permits.loc[1:10]
sf_sub
# comes directly after it and then replacing any remaining NaN's with 0
sf_nona=sf_sub.fillna(0)
sf_nona=sf_sub.fillna(method="bfill",axis=0).fillna(0)
sf_nona.head()
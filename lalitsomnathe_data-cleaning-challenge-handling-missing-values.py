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
sf_permits.sample(5) # missing data at Street Number Suffix column
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()
#missing_values_count
# look at the # of missing points in the first ten columns
print(missing_values_count[0:10])
print(nfl_data.shape)
#checking for which columns we have 1 or more NaN values
nfl_data.columns[nfl_data.isnull().any()]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing
missing_values_count_sf=sf_permits.isnull().sum()
# missing_values_count_sf.sum()
total_cells_sf=np.product(sf_permits.shape)
total_missing_sf=missing_values_count_sf.sum()

# percent of data that is missing in sf_permits
(total_missing_sf/total_cells_sf) * 100
# look at the # of missing points in the first ten columns
missing_values_count.sort_values(ascending=False).head() # TOP 5 missing value counts and their columns

missing_values_count[0:10]
sf_permits.isnull().sum().sort_values(ascending=False).head(5)
# remove all the rows that contain a missing value
nfl_data.dropna()
nfl_data.shape
# remove all columns with at least one missing value

columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()

# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits.dropna()
# Now try removing all the columns with empty values. Now how much of your data is left?
columns_with_na_dropped_sf = sf_permits.dropna(axis=1)
#columns_with_na_dropped_sf.head()

print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped_sf.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data

# replace all NA's with 0
subset_nfl_data.fillna(0)
#subset_nfl_data 
# inplace is False, hence change is not reflected
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then 
subset_sf_permits=sf_permits.loc[:, 'Existing Construction Type':'Record ID'].head(5
                                                                                )
subset_sf_permits.head(5)
#subset_sf_permits.isnull().sum()
subset_sf_permits.fillna(method='bfill',axis=0).fillna(0)
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
total_vals = np.product(sf_permits.shape)
nulls = sf_permits.isnull().sum()
total_nulls = nulls.sum()

percentage = (total_nulls/total_vals)*100
print('Total missing values in SF permit: ', percentage)
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
print(sf_permits[['Street Number Suffix', 'Zipcode']].sample(10))
#Looks like all values in street number sfuffix are null
#Lets check with the null counts of the column
(nulls['Street Number Suffix']/sf_permits.shape[0])*100
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
sf_perm_dropcols = sf_permits.dropna(axis=1)
sf_perm_dropcols.head()
#Lets look at how mNy columns we lost
print('Columns in the original', sf_permits.shape[1])
print('Columns in the new', sf_perm_dropcols.shape[1])
print('Percentage of columns lost: ', ((sf_perm_dropcols.shape[1]/sf_permits.shape[1])*100), '%')
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
sf_perm = sf_permits.copy()
sf_permits.fillna(method = 'bfill', axis = 0).fillna('0').head()

## This is a "fast and dirty" way to do this and does not look like a good method! 
# As we can see that even all values in the Street Number Suffix get replaced by the value after it and this makes no sense!
#importing the data
address_db = pd.read_csv("../input/openaddresses-us-west/ca.csv")
sf_perm = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")
address_db.sample(10)
sf_perm.head()
#Let's try to look for the City San Francisco of SF
print(address_db[address_db['CITY'] == 'SAN FRANCISCO'].size)
address_db['CITY'].value_counts()
address_db['DISTRICT'].unique()
address_db['REGION'].unique()

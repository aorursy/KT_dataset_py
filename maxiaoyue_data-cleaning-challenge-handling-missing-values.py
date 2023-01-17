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
sf_permits.head()
sf_permits.tail()
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
sf_permits.isnull().sum()[:10]
# your turn! Find out what percent of the sf_permits dataset is missing
sf_permits.isnull().sum().sum() * 100.0/np.prod(sf_permits.shape)
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
sf_permits[['Street Number Suffix', 'Zipcode']].isnull().sum()
sf_permits.shape
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
print (sf_permits.shape[0])
print (sf_permits.dropna().shape[0])
col_na_dropped_sf_permits = sf_permits.dropna(axis= 1)
print ('Columns in original dataset: %d' % sf_permits.shape[1])
print ('Columns with missing values dropped: %d' % col_na_dropped_sf_permits.shape[1])
# Now try removing all the columns with empty values. Now how much of your data is left?
sf_permits.dropna(axis = 1, how = 'all').shape[1]
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
sf_permits.fillna(method = 'bfill', axis = 0).fillna(0)
sf_permits[sf_permits.Zipcode.isnull()]['Block'].head()
sf_permits.sort_values(by = 'Block')[['Block','Zipcode']].loc[sf_permits.Block == '0552']
sf_permits.groupby('Block').Zipcode.nunique().shape
mask = (sf_permits.groupby('Block').Zipcode.nunique() > 1)
sf_permits.groupby('Block').Zipcode.nunique()[mask].head()
sf_permits.Zipcode.isnull().sum()
sf_permits_Zip_na_fill_with_block = sf_permits.sort_values(by = 'Block').Zipcode.fillna(method = 'bfill').fillna(method = 'ffill')
sf_permits_Zip_na_fill_with_block.isnull().sum()
sf_permits.columns
sf_permits.sort_values(by= ['Street Name','Street Number']).Zipcode.head()
sf_permits_Zip_na_fill_with_street = sf_permits.sort_values(by= ['Street Name','Street Number']).Zipcode.fillna(method = 'bfill').fillna(method = 'ffill')
sf_permits_Zip_na_fill_with_street.isnull().sum()
(sf_permits_Zip_na_fill_with_street.sort_index() != sf_permits_Zip_na_fill_with_block.sort_index()).sum()
mask = (sf_permits_Zip_na_fill_with_street.sort_index() != sf_permits_Zip_na_fill_with_block.sort_index())
sf_permits_Zip_na_fill_with_street[mask]
sf_permits_zip_na_fill_with_block_street = sf_permits.sort_values(by = ['Block', 'Street Name', 'Street Number']).Zipcode.fillna(method = 'bfill').fillna(method = 'ffill')
sf_permits_zip_na_fill_with_block_street.sort_index().head()
sf_permits_Zip_na_fill_with_block.sort_index().head()
(sf_permits_zip_na_fill_with_block_street.sort_index() != sf_permits_Zip_na_fill_with_block.sort_index()).sum()
(sf_permits_zip_na_fill_with_block_street.sort_index() != sf_permits_Zip_na_fill_with_street.sort_index()).sum()
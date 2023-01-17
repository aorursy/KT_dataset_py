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
sf_permits.head(10)
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
print (sf_permits.shape)
sf_permits_nan = sf_permits.isnull().sum()
print (sf_permits_nan.loc[lambda x: x != 0])
print (sf_permits.isnull().sum().sum())
print (np.product(sf_permits.shape))
print ('Percentage of NaNs in total dataset %.3f' %(sf_permits.isnull().sum().sum() * 100 / np.product(sf_permits.shape)))
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
sf_columns = sf_permits.columns
print (sf_columns[4:8])
print (sf_columns[-3])
sf_permits_slice = sf_permits.iloc[:, np.r_[4:8, -3]]
print (sf_permits_slice.head())
print (sf_permits_slice[sf_permits_slice['Street Number Suffix'].notnull()].head())
print (sf_permits_slice['Street Number Suffix'][sf_permits_slice['Street Number Suffix'].notnull()].unique())
print (sf_permits_slice[sf_permits_slice['Zipcode'].isnull()].head())
def print_zips(street_num, block):
    print (sf_permits_slice[(sf_permits_slice['Street Number'] == street_num) & (sf_permits_slice['Block'] == block)])
print_zips(2550, '0552')
print_zips(1235, '0779')
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
sf_permits.dropna(axis=1)
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
print (sf_permits_slice.head())
print (sf_permits_slice.info())
sf_permits_slice['Street Number'] = sf_permits_slice['Street Number'].astype(int)
sf_permits_slice.sort_values(by=['Street Number', 'Block', 'Lot'], inplace=True) #I don't know what Block and Lot mean in SF, so just a guess
print (sf_permits_slice.head())
sf_permits_slice['Zipcode'] = sf_permits_slice['Zipcode'].fillna(method='bfill', axis=0).fillna(0)
print (sf_permits_slice.head())
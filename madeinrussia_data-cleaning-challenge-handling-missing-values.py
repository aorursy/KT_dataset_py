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
sf_permits.sample(10)

sf_permits.columns
sf_permits.columns[sf_permits.isnull().any()].tolist()
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
sf_missing_values_count = sf_permits.isnull().sum()
sf_missing_values_count[sf_missing_values_count>0]
sf_total_cells = np.product(sf_permits.shape)
sf_total_missing = sf_missing_values_count.sum()

# percent of data that is missing
(sf_total_missing/sf_total_cells) * 100
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
(sf_permits['Zipcode'].isnull() != sf_permits['Location'].isnull()).sum()
pd.set_option('max_columns', 50) #we want to see more columns in table to have a better idea about contents of records without ZipCode
sf_permits.loc[sf_permits['Zipcode'].isnull()] #looking for empty zipcodes to check out what else we have inside
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
fl_columns_with_na_dropped = sf_permits.dropna(axis=1)
fl_columns_with_na_dropped.head()
print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % fl_columns_with_na_dropped.shape[1])
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
fl_columns_with_na_filled = sf_permits.fillna(method = 'bfill', axis=0).fillna(0)
fl_columns_with_na_filled.head()
# We will save the flags for features that was inputed automatically 
sf_permits_plus = sf_permits.copy()

cols_with_missing = (col for col in sf_permits.columns 
                                 if sf_permits[col].isnull().any())
for col in cols_with_missing:
    sf_permits_plus[col + '_was_missing'] = sf_permits_plus[col].isnull()

#splitting objects and numbers for inputting purposes

sf_permits_nums = sf_permits_plus.select_dtypes(exclude=['object'])
sf_permits_obj = sf_permits_plus.drop(sf_permits.select_dtypes(exclude=['object']).columns, axis = 1)

#Try to input all the data with median strategy for the numeric ones
from sklearn.preprocessing import Imputer
my_imputer = Imputer(strategy='median')
sf_permits_nums = pd.DataFrame(my_imputer.fit_transform(sf_permits_nums), index=sf_permits_nums.index, columns=sf_permits_nums.columns)
sf_permits_obj = sf_permits_obj.fillna(method = 'bfill', axis=0).fillna(0)
sf_permits_filled = pd.concat([sf_permits_obj, sf_permits_nums])
sf_permits_filled.head()
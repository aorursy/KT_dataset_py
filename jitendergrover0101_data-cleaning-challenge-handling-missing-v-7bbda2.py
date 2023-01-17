# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
Football_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
building_data = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 
print(Football_data.columns)
print(Football_data.dtypes)
print(Football_data.head())
print(Football_data)

print('************************************************************')
print('************************************************************')
print('Building dataset start')
print(building_data.columns)
print(building_data.dtypes)
print(building_data.head())
print(building_data)
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
Football_data.sample(5)
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?
# set seed for reproducibility


building_data.sample(5)
# your code goes here :)
# get the number of missing data points per column
missing_values_count = Football_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(Football_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing
# get the number of missing data points per column in building
building_missing_values_count = building_data.isnull().sum()

# look at the # of missing points in the first ten columns
building_missing_values_count[0:10]

# how many total missing values do we have?
building_total_cells = np.product(building_data.shape)
building_total_missing = building_missing_values_count.sum()

# percent of data that is missing
(building_total_missing/building_total_cells) * 100
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
building_missing_values_count[0:]
# remove all the rows that contain a missing value
Football_data.dropna()
print(Football_data)
# remove all columns with at least one missing value
columns_with_na_dropped = Football_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % Football_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
building_data.dropna()
# Now try removing all the columns with empty values. Now how much of your data is left?
building_columns_with_na_dropped = building_data.dropna(axis=1)
building_columns_with_na_dropped.head()

print("Building Columns in original dataset: %d \n" % building_data.shape[1])
print("Building Columns with na's dropped: %d" % building_columns_with_na_dropped.shape[1])
# get a small subset of the NFL dataset
#subset_Football_data = Football_data.loc[:, 'EPA':'Season'].head()
subset_Football_data = Football_data.loc[:, 'EPA':'Season']
subset_Football_data
# replace all NA's with 0
subset_Football_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_Football_data.fillna(method = 'bfill', axis=0).fillna(0)
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing any remaining NaN's with 0

building_subset_Football_data = building_data.loc[:, 'Permit Number':'Street Suffix']
building_subset_Football_data

building_subset_Football_data.fillna(method = 'bfill', axis=0).fillna(0)
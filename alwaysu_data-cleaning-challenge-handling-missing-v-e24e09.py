# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
#nfl_data.sample(5)
nfl_data.head()
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?
sf_permits.head()
# your code goes here :)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
# .shap means (num of rows , num of columns)
# np.product means the multiply of num of rows and num of columns
# print(nfl_data.shape)
 #print(np.product((1, 2)))
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing
missing_value_my = sf_permits.isnull().sum()
missing_value_my[0:10]

total_num_my = np.product(sf_permits.shape)
total_missing_value_my = missing_value_my.sum()

percentage_missing = total_missing_value_my/total_num_my
percentage_missing

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
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
sf_permits.dropna(axis=1).head()
Removed_my = sf_permits.dropna(axis=1)
print(Removed_my.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season']
subset_nfl_data.head()
# replace all NA's with 0
subset_nfl_data.head().fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
#subset_nfl_data.fillna(method = 'bfill', axis=0)
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0).head()
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing any remaining NaN's with 0
sf_permits.fillna(method = 'bfill', axis=0).fillna(0).head()
#sf_permits.head()
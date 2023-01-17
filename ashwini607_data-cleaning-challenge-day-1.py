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
# your code goes here :)
#yep! I am able to see the missing data
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
missing_value_count2 = sf_permits.isnull().sum()
total_cell2 = np.product(sf_permits.shape)
total_missing2 = missing_value_count2.sum()
(total_missing2/total_cell2)*100
# look at the nfl_data of missing points in the first ten columns
missing_values_count[0:10]
sf_permits["Street Number"].sample (10)
count_null_in_streetNumber = sf_permits["Street Number"].isnull().sum()
print ("null value in street number")
print(count_null_in_streetNumber)
count_null_in_zipcode = sf_permits["Zipcode"].isnull().sum()
print("null value in zipcode column")
print(count_null_in_zipcode)
print("column size of street number")
len(sf_permits["Street Number"])
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
column_with_dron_na = sf_permits.dropna(axis=1)
print("columns with na value in sf dataset: %d \n" % sf_permits.shape[1])
print("columns without na value in sf dataset: %d \n" % column_with_dron_na.shape[1])
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
subset_sf_permits = sf_permits.loc[:, 'Permit Number':'Street Name'].head()
subset_sf_permits
subset_sf_permits.fillna(0)
subset_sf_permits.fillna(method = 'bfill', axis=0).fillna(0)

# I am done for today. It was a good exercise.
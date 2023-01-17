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
sf_permits.sample(3)
# your code goes here :)
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
# get the number of missing data points per column
sf_missing_count = sf_permits.isnull().sum()
print(sf_missing_count)
# look at the # of missing points in the first ten columns
sf_missing_count[0:10]
# total missing values in sf_permits dataset
sf_total_cell=np.product(sf_permits.shape)
print(sf_permits.shape)
#Sum of missing values in sf_permits dataset
tot_sf_missing=sf_missing_count.sum()
print(tot_sf_missing)

# Percentage of missing values
print("Missing values are" , round(tot_sf_missing*100/sf_total_cell),"%")


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
#Drops all the column
# Now try removing all the columns with empty values. Now how much of your data is left?
sf_columns_with_na_dropped=sf_permits.dropna(axis=1)
sf_columns_with_na_dropped.head()

#Print the data for total columns and without missing values
print("Columns in original dataset %d \n" %sf_permits.shape[1])
print("Columns after Na's are removed %d" %sf_columns_with_na_dropped.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
sample_sf_permits=sf_permits.loc[:,'Structural Notification': 'Proposed Units'].head(10)
sample_sf_permits
#Replacing NaN's with 0 and date columns with a default date= 07/13/2018 :)
values = {'Permit Expiration Date': '07/13/2018', 'Existing Use': 'Not Aavailable', 'Proposed Use': 'Not Aavailable'}
sample_sf_permits.fillna(method='bfill',axis=0).fillna(value=values).fillna(0)
# comes directly after it and then replacing any remaining NaN's with 0
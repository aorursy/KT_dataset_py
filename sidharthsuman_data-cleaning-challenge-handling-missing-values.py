# modules we'll use
import pandas as pd
import numpy as np
from pprint import pprint
# read in all our data
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
nfl_data.sample(5)
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?
display(sf_permits.columns )
sf_permits.head()
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
# your turn! Find out what percent of the sf_permit dataset is missing

#Number of data points with missing values in our dataset
missing_values_sf_permits = sf_permits.isnull().sum()
#
pprint(missing_values_sf_permits)

#Total number of cells in our dataset
total_cells_sf_permits = np.product(sf_permits.shape)
total_missing_sf_permits = missing_values_sf_permits.sum()

#Percentage of missing data 
sf_permits_missing_percent = (total_missing_sf_permits/total_cells_sf_permits)*100
print('Percentage of missing data in San Francisco building permits dataset: {0}'.format(sf_permits_missing_percent))
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
print(sf_permits.shape)
print("Missing values in Street Number Suffix: ",missing_values_sf_permits["Street Number Suffix"])
print("Missing values in Zipcode: ",missing_values_sf_permits["Zipcode"])
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
columns_with_na_dropped_sf_permits = sf_permits.dropna(axis=1)
print(columns_with_na_dropped_sf_permits.shape)
display(columns_with_na_dropped_sf_permits.head())
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

sf_permits.fillna(method = 'bfill',axis=0).fillna('0').head()
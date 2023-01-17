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
sf_permits.head()
# yes there are a lot of missing values in the Street Number Suffix and Site Permit.
#Some values are missing in columns Proposed Construction Type and Proposed Constructor Type Description.  
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
sf_permits_missing_vals = sf_permits.isnull().sum()
sf_permits_missing_vals[0:20]
total_cells_sf_permits = np.product(sf_permits.shape)
total_missing_vals = sf_permits_missing_vals.sum()
sf_permits_missing_data_percentage = (total_missing_vals/total_cells_sf_permits)*100
print("Total missing values percentage in the sf_permits dataset is {0}".format(sf_permits_missing_data_percentage))
# more than quarter of the data is missing in the sf_permits dataset. 
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
sf_permits.iloc[:,[7,40]]
# I think the zipcode exists but wasn't recorded as and the Street Number Suffix are missing because they don't exist for these records. 
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
print("# of rows before dropping missing values {0}".format(sf_permits.shape[0]))
sf_permits_without_null = sf_permits.dropna()
print("# of rows after dropping missing values {0}".format(sf_permits_without_null.shape[0]))
print("All of the rows are dropped if we use dropna() on the entire dataset")
# Now try removing all the columns with empty values. Now how much of your data is left?
print("Columns before dropping the columns with any NA values {0}".format(sf_permits.shape[1]))
sf_permits_with_dropped_na = sf_permits.dropna(axis=1)
sf_permits_with_dropped_na.head()
print("Columns after dropping the columns with any NA values {0}".format(sf_permits_with_dropped_na.shape[1]))
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
sf_permits = sf_permits.fillna(method = 'bfill',axis = 1).fillna("0")
sf_permits.head(20)
missing_sf_permits_after_fill = sf_permits.isnull().sum()
print("Missing values in sf_permits after fill = ", missing_sf_permits_after_fill.sum())
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

sf_permits.sample(100)
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
total_cells_sf = np.product(sf_permits.shape)
missing_value_sf = sf_permits.isnull().sum()
total_missing_sf = missing_value_sf.sum()
(total_missing_sf/total_cells_sf)*100
# look at the # of missing points in the first ten columns
missing_value_sf

# Street number suffix may or may not exist. Therefore, not all missing values may represent lack of records. 
# On the other hand, absence of Zipcodes is unlikely, and hence missing values probably arise because the data was not recorded.
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
rows_na_dropped_sf = sf_permits.dropna(axis=0)
rows_na_dropped_sf.head()
# Apparently, all rows of sf_permits had at least one missing value, and hence the data set resulting from NA removal from rows left us with
# a null data set

# Now try removing all the columns with empty values. Now how much of your data is left?
cols_na_dropped_sf = sf_permits.dropna(axis=0)
cols_na_dropped_sf.head()
# Same issue with removal of NAs from columns in sf_permits.
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then 

sf_permits.fillna(method = 'ffill', axis=0).fillna("0")

# I have not subsetted the sf_permits data
# I have filled everything that comes after the first value after the NA in each row with a 0, as per the previous example.
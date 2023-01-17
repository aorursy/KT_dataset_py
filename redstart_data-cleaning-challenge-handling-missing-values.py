# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
nfl_data.sample(10)
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?

# Useful attributes / functions
sf_permits.columns  # get the column names of the dataframe
sf_permits.shape    # get number of rows and columns of dataframe
sf_permits.info()   # get an overview of the dataframe, see also https://stackoverflow.com/questions/13921647/python-dimension-of-data-frame/47139464#47139464

sf_permits.sample(10)

# Result:
# Yes, there is missing data (NaN values)
# Some features seem to be missing altogether, e.g. 'Street Number Suffix' or 'Site Permit'.
# get the number of missing data points per column
nfl_missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
nfl_missing_values_count[0:10]
# how many total missing values do we have?
nfl_total_cells = np.product(nfl_data.shape)
nfl_total_missing = nfl_missing_values_count.sum()

# percent of data that is missing
(nfl_total_missing/nfl_total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing

# get the number of missing data points per column
sf_missing_values_count = sf_permits.isnull().sum()
    # isnull() returns a boolean dataframe of the same size as the input data
    # returns a Series object with the number of missing values for each column of the input dataframe

# look at the # of missing points in the first ten columns
sf_missing_values_count[0:10]

# how many total missing values do we have?
sf_total_cell_number = np.product(sf_permits.shape)
sf_total_missing_number = sf_missing_values_count.sum()

# percent of data that is missing
(sf_total_missing_number/sf_total_cell_number) * 100

# Appr. one quarter of the dataframe are missing values

# look at the # of missing points in the first ten columns
nfl_missing_values_count
# look at the # of missing points in all nonzero columns sorted descending. 
# I want to see all rows and not have my output clipped
pd.set_option('display.max_rows', 1000)
sf_missing_values_count[sf_missing_values_count > 0].sort_values(ascending=False)


# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values.
# How many are left?
sf_dropna_rows = sf_permits.dropna()
sf_dropna_rows
# Now try removing all the columns with empty values. Now how much of your data is left?
sf_permits_dropna_columns = sf_permits.dropna(axis=1)
print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % sf_permits_dropna_columns.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head() # take a slice of the original dataframe
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the remaining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing any remaining NaN's with 0
sf_permits.fillna(method = 'bfill', axis=0).fillna(0)
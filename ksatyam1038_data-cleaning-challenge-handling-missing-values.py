# modules we'll use
import pandas as pd
import numpy as np

# read in all our datad

nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 
sf_permits.sample(5)
#lets count total number of cell in dataframe
totalcells=np.product(sf_permits.shape)
# get the number of missing data points per column
missing_values_count = sf_permits.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# percent of the sf_permits dataset is missing
total_cells = np.product(sf_permits.shape)
total_missing= missing_values_count.sum()

#percentage of data that is missing
(total_missing/total_cells)*100
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# remove all the rows that contain a missing value
sf_permits.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = sf_permits.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# get a small subset of the permits  dataset
subset_sf_permits = sf_permits.loc[:, 'Permit Number':'Street Suffix'].head()
subset_sf_permits


# replace all NA's with 0
subset_sf_permits.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_sf_permits.fillna(method = 'bfill', axis=0).fillna(0)
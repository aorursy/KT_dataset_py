# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 

#sample 5 rows from the nfl_data file. Already many NaNs
nfl_data.sample(5)
#sample from sf_permits dataset. Hadfull of NaNs likewise. 
sf_permits.sample(5)
#get the number of missing data points per column
missing_val_count=nfl_data.isnull().sum()
#look at the number of missing values in the first ten columns
missing_val_count[0:10]
# how many total missing values do we have?
total_cells=np.product(nfl_data.shape)
total_missing=missing_val_count.sum()

#percentage of data that is missing
(total_missing/total_cells)*100

missing_sf_count=sf_permits.isnull().sum()
missing_sf_count[0:10]
#many missing data in some columns too
#find the percentage of misising values by deviding total cells by missing cells
total_cells_sf=np.product(sf_permits.shape)
total_missing_sf = missing_sf_count.sum()

#percentage of data that is missing
(total_missing_sf/total_cells_sf)*100

#Here over 25% of the observations is missing
# remove all the rows that contain a missing value
nfl_data.dropna()

#it dropped all of the data, because every row had an NaN value. Try dropping columns with NaNs instead:
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
#clean NaN from sf_permit dataset
sf_permits.dropna()
#again all the data is dropped
#Try dropping columns with NaNs instead:
# remove all columns with at least one missing value
columns_with_na_dropped_sf = sf_permits.dropna(axis=1)
columns_with_na_dropped_sf.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" %sf_permits.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped_sf.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data

# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
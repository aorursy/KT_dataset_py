# importing libraries
import pandas as pd
import numpy as np

#importing datasets
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 
nfl_data.sample(5)
#looking for missing values in second dataset
sf_permits.sample(20)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()
#Looking at first few columns
missing_values_count[0:10]
#to calculate missing values measure in total data
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
missing_values_count_sf = sf_permits.isnull().sum()
missing_values_count_sf[0: 10]
#Percent for sf_permit
total_cells_sf = np.product(sf_permits.shape)
total_missing_sf = missing_values_count_sf.sum()

#percent of data that is missing
(total_missing_sf/total_cells_sf)*100
# looking at the # of missing points in the first ten columns
missing_values_count[0:10]
# remove all the rows that contain a missing value
nfl_data.dropna()
# removing all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
#Now removing data from sf_permits
sf_col_after_drop = sf_permits.dropna(axis = 1)
sf_col_after_drop.head()
print("Columns without drop: %d" %sf_permits.shape[1])
print("Columns after drop: %d" %sf_col_after_drop.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the remaining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
#Replacing all the NaN's in the sf_permit data with the one that
# comes directly after it and then 
sf_permits.fillna(0).head()
sf_permits.fillna(method = 'bfill', axis = 1).head()
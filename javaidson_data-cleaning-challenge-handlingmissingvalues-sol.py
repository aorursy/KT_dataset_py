import pandas as pd

import numpy as np



sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

np.random.seed(0) 

sf_permits.sample(5)
# get the number of missing data points per column

missing_values_count = sf_permits.isnull().sum()



# look at the # of missing points in the first ten columns

missing_values_count[0:10]
# how many total missing values do we have?

total_cells = np.product(sf_permits.shape)

total_missing = missing_values_count.sum()



# percent of data that is missing

(total_missing/total_cells) * 100
# look at the # of missing points in the first ten columns

missing_values_count[0:10]
# remove all the rows that contain a missing value

sf_permits.dropna()



# We left will zero rows :|
# remove all columns with at least one missing value

columns_with_na_dropped = sf_permits.dropna(axis=1)

columns_with_na_dropped.head()
# just how much data did we lose?

print("Columns in original dataset: %d \n" % sf_permits.shape[1])

print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# get a small subset of the sf_permits dataset

subset_sf_permits = sf_permits.loc[:, 'Permit Number':'Zipcode'].head()

subset_sf_permits
# replace all NA's with 0

subset_sf_permits.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 

# then replace all the reamining na's with 0

subset_sf_permits.fillna(method = 'bfill', axis=0).fillna(0)
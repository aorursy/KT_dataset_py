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

sf_permits.sample(10)
#units, unit suffix, streed number suffix are all missing.plus a ton more
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape) #Multiplies the number of rows and columns from the shape attribute
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing
sf_permits.isnull().sum().sum()/np.product(sf_permits.shape) *100
#26% of missing data
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
sf_permits
sf_permits[sf_permits['Street Number Suffix'].notnull()]
#only 2216 of our rows have a street number suffix. A lot of street number suffixes are optional.
#I would say that this one has missing values because it doesn't exist

sf_permits[sf_permits['Zipcode'].isnull()]
#all locations have a zip code in America so for this we can assume that these values were because they weren't recorded.

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
#same with the nfl dataset, none of our data is left!
# Now try removing all the columns with empty values. Now how much of your data is left?
sf_permits.dropna(axis=1)
print(np.product(sf_permits.shape))
print(np.product(sf_permits.dropna(axis=1).shape))
#We lost a ton of data it seems. but let's check how many of this difference were nulls
sf_permits.isnull().sum().sum()
#So we lost over half our data from this!!
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
#there is also forward fill which goes the opposite way
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing any remaining NaN's with 0
sf_permits.fillna(method='bfill').fillna(0)
#if you could find a dataset with zip code information tied to city,
#then you could left join the sf_permits dataframe to it at the missing values

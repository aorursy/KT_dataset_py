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
# your turn! 
sf_permits.sample(5)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! 

# in the previous cell we got total percentage of missing values: 
100 * sf_permits.isnull().sum().sum() / np.product(sf_permits.shape)

# before handling missing values, I prefer to fill empty and NaNs values with NaN (if any):
sf_permits = sf_permits.fillna(np.nan)

# Viewing percentage in all columns helps us to understand which columns should be removed and where we can restore the missing values:
( 100 * sf_permits.isnull().sum() / sf_permits.shape[0] ).sort_values(ascending=False)

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
# Now try removing all the columns with empty values. Now how much of your data is left?
print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % sf_permits.dropna(axis=1).shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
# Your turn! 


# in the previous cells we got data subset to fill missing data with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")

# here is a code to get subset and fill missing data with with mean/median values in it (with pandas functions only, without sklearn.preprocessing.Imputer)
columns_to_restore = ( (sf_permits.isnull().sum()/sf_permits.shape[0] > 0) & (sf_permits.isnull().sum()/sf_permits.shape[0] < .5) ).tolist()     # select columns' indexes with <50% of missing data
subset_sf_permits = (sf_permits
                     .loc[:, columns_to_restore]
                     .select_dtypes(include=[np.number])
                     .apply(lambda x: x.fillna(x.median()))
                    )
subset_sf_permits.head(10)
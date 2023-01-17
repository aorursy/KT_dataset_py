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
import numpy as np
import pandas as pd
sf_permits=pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")
np.random.seed(0) 
sf_permits.sample(6)

# your turn! Find out what percent of the sf_permits dataset is missing
missing_values_count=sf_permits.isnull().sum()
total_cells=np.product(sf_permits.shape)
total_missing=missing_values_count.sum()
(total_missing/total_cells)*100

#your turn! * Look at the columns `Street Number Suffix` and `Zipcode` from the `sf_permits` datasets. Both of these contain missing values.
#Which, if either, of these are missing because they don't exist? Which, if either, are missing because they weren't recorded?
missing_values_count[0:45]
#(columns `Street Number Suffix`=these contains almost missing values(NaN),so, it's look like not recorded and
#other `Zipcode`=exist but contains less missing values)

# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left? 
sf_permits.dropna()
#(zeros are left)

# Now try removing all the columns with empty values. Now how much of your data is left?
columns_with_na_dropped=sf_permits.dropna(axis=0)
columns_with_na_dropped.head()
#(43 are left)
print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
#( 43 columns)

# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing any remaining NaN's with 0

# get a small subset of the sf_permit dataset
subset_sf_permits=sf_permits.loc[:,'Street Number Suffix':'Zipcode'].head()
subset_sf_permits
# replace all NA's with 0
subset_sf_permits.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_sf_permits.fillna(method ='bfill',axis=0).fillna(0)



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

# Now try removing all the columns with empty values. Now how much of your data is left?

# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing any remaining NaN's with 0
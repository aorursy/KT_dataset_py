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
# your turn! Find out what percent of the sf_permits dataset is missing
sf_missing_values_count = sf_permits.isnull().sum()
sf_total_missing = sf_missing_values_count.sum()
sf_total_cells = np.product(sf_permits.shape) #.shape returns #rows and #columns of the array
# print(np.product([3,3]))
(sf_total_missing/sf_total_cells)*100 # percent of data that is missing
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
#columns_with_na_dropped = nfl_data.dropna(1) // same as above
columns_with_na_dropped.head() #.head() function --> default is to return top 5 rows.  Specify int for more/fewer rows.
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1]) 
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
#% are used for formatting strings. 
#%s acts a placeholder for a string; %d acts as a placeholder for a number. 
#Their associated values are passed in via a tuple using the % operator.
#\n prints a return line
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits.dropna()
# Now try removing all the columns with empty values. Now how much of your data is left?
sf_permits_columns_with_na_dropped = sf_permits.dropna(axis=1)
sf_permits_columns_with_na_dropped.head(2)

print("Columns in original dataset: %d \n" % sf_permits.shape[1])
#If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n.
print("Columns with na's dropped: %d" % sf_permits_columns_with_na_dropped.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
#.loc[] is a purely label-location based indexer for selection by label.
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
# method : {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}, default None
#    Method to use for filling holes in reindexed Series 
#    pad / ffill: propagate last valid observation forward to next valid 
#    backfill / bfill: use NEXT valid observation to fill gap
# axis : {0 or ‘index’, 1 or ‘columns’}
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing all the na's with 
sf_permits.fillna(method='bfill', axis=0).fillna("0")
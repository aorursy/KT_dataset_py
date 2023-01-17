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
missing_count = sf_permits.isna().sum()
total_missing = missing_count.sum()
total_fields = np.product(sf_permits.shape)

"Proportion of missing values in Building Permit issued in S.F.: {0:.2f}%".format(total_missing / total_fields * 100)
# Let's figure out which are the fields with more data missed
miss_by_field = missing_count.loc[missing_count>0].sort_values(ascending=False)

miss_by_field.plot.bar()

# Propotion of the missing values by field
total_rows = len(sf_permits)
miss_by_field.map(lambda x: x / total_rows * 100).plot.bar()
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# a helper function
def list_methods(o):
    return ', '.join([i for i in dir(o) if i[0] != '_'])

list_methods(sf_permits)
column_names = ['Zipcode', 'Street Suffix']

zipcode_missing = sf_permits['Zipcode'].isna()

# list the rows where zipcode is missing
sf_permits[zipcode_missing]

# zipcode missing values seem that were NOT recorded, because location also is missing
st_suffix_missing = sf_permits['Street Suffix'].isna()
# list the rows where suffix is missing
sf_permits[st_suffix_missing]

# the street suffix missings seem to be the information does NOT exists
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
sf_drop = sf_permits.dropna(axis=1)

print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % sf_drop.shape[1])
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
sf_permits.fillna(method='bfill', axis=0).fillna(0)
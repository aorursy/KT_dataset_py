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
# your code goes here :)
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
permit_cells = np.product(sf_permits.shape)
permit_missing_cells = np.sum(sf_permits.isnull().sum())

(permit_missing_cells/permit_cells)*100
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
permit_col_missing = sf_permits.isnull().sum()
permit_col_missing

# Every place has a zipcode to it, so that data is probably not recorded and we should try to figure it out. There is a possibility that
# street number suffix doesnt exist, so we can probably leave these alone.
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits.dropna().shape[0] 
# looks like we would also lose all our rows if we follow this strategy :(
# Now try removing all the columns with empty values. Now how much of your data is left?
na_columns_removed = sf_permits.dropna(axis = 1)
na_columns_removed.shape[1]
print("Total columns in raw data: %d \n" %sf_permits.shape[1])
print("Total columns removed from data: %d \n" %na_columns_removed.shape[1])
print("Percentage of data removed: %.2f \n" %(na_columns_removed.shape[1]*100/sf_permits.shape[1]))
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's with the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing any remaining NaN's with 0
sf_permits.fillna(method = 'bfill', axis = 0).fillna(0).head()
# Lets take a look at what other columns are available in the dataset that can help me decipher zipcode.
sf_permits.columns
# If i have to work with the first strategy - using 'Location' to get 'Zipcode', ideally i want each location to be associated with one
# Zipcode. We can check this by getting a count of unique Zipcodes for each Location.
# First lets check a few values in the location column
sf_permits.loc[:,'Location'].head()
pd.pivot_table(data = sf_permits, values = "Zipcode", index = "Location", aggfunc = pd.Series.nunique, dropna=False).sort_values("Zipcode", ascending = False).head()
# the total missing zipcodes are 1716, obtained using the code below:
sf_permits.loc[:,'Zipcode'].isnull().sum()

# do we even have Location values for these rows? Apparently we do only for 16 rows.
missing_zipcodes = sf_permits.loc[:, 'Zipcode'].isnull()
sf_permits.loc[missing_zipcodes,'Location'].value_counts(dropna = False)
sf_permits.loc[:, "Block":"Street Number Suffix"].head()
sf_permits_copy = sf_permits
sf_permits_copy['Street_Identifier'] = sf_permits_copy.loc[:,"Block":"Street Number Suffix"].astype(str).apply(lambda x: '-'.join(x), axis = 1)
sf_permits_copy.loc[:, 'Street_Identifier'].head()
pd.pivot_table(data= sf_permits_copy, index = ['Street_Identifier'], values = 'Zipcode', aggfunc = pd.Series.nunique).\
sort_values('Zipcode', ascending = False).head()
# get a list of each identifier with its corresponding Zipcode
zipcode_master = sf_permits_copy.loc[-missing_zipcodes, ['Street_Identifier', 'Zipcode']].drop_duplicates()

# get a list of identifiers from rows where Zipcode is missing
missing_identifiers = sf_permits_copy.loc[missing_zipcodes, 'Street_Identifier'].drop_duplicates()

# Before we merge, lets check if any identifier from the missing rows is even present in the zipcode_master.
np.sum(missing_identifiers.apply(lambda x: x in zipcode_master['Street_Identifier']))

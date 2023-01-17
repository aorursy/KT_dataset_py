# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv",low_memory=False)
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv",low_memory=False)

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
sfmissing_values_count = sf_permits.isnull().sum()
sftotal_cells = np.product(sf_permits.shape)
sftotal_missing = sfmissing_values_count.sum()

# percent of data that is missing
print((sftotal_missing/sftotal_cells) * 100, "percentage of dataset is missing.")
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
#this leaves us with 0 rows
# Now try removing all the columns with empty values. Now how much of your data is left?
sfcolumns_with_na_dropped = sf_permits.dropna(axis=1)
print("Number of columns remaining after na's dropped: %d" % sfcolumns_with_na_dropped.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
#Create a copy of the sf_permits dataframe
sf_permits_copy=sf_permits.copy()
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then 
sf_permits.fillna(method = 'bfill', axis=0).fillna("0").head(5)
#Import san francisco address dataset to fill null in Zipcode
import pandas as pd
san_francisco_address = pd.read_csv("../input/openaddress-sanfrancisco/san_francisco.csv")
#Get number of observations will null in Zipcode
print("Total number of nulls in  Zipcode: ",sum(pd.isnull(sf_permits_copy['Zipcode'])))
sf_permits_copy['Street Name'] = sf_permits_copy['Street Name'].apply(lambda x: x.upper())
sf_permits_copy['Street Number'] = sf_permits_copy['Street Number'].apply(str)
san_francisco_address['STREET'] = san_francisco_address['STREET'].apply(str)
index_nullzips = sf_permits_copy["Zipcode"].isnull()
sf_permits_copy.loc[index_nullzips,"Zipcode"]=san_francisco_address.loc[sf_permits_copy[index_nullzips].index,"POSTCODE"]
sf_permits_copy.reset_index(inplace=True)
san_francisco_address.reset_index(inplace=True)
#Get number of observations with null in Zipcode
print("Total number of nulls in Zipcode after replacing: ",sum(pd.isnull(sf_permits_copy['Zipcode'])))
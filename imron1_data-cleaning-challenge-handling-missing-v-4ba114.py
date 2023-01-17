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
sf_permits2=pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv",header=None)
sf_permits2.head()
sf_permits2.shape
sf_permits2.describe()
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permit dataset is missing
# get the number of missing data points per column
missing_values_countx = sf_permits.isnull().sum()
print ("Missing values count are:\n")
print (missing_values_countx)
# look at the # of missing points in the first ten columns
print ("")
print ("Missing values in the first 10 column :\n")
print (missing_values_countx[0:10])

# Calculate total number of missing values
totalMissing = missing_values_countx.sum()
print ("")
print ("Calculate total number of missing values :")
print (totalMissing)


# how many total missing values do we have?
total_cells = np.product(sf_permits2.shape)
total_missing = totalMissing

# percent of data that is missing
percentage_data_missing=(total_missing/total_cells) * 100
percentage_data_missing_round=round(percentage_data_missing,3)
print ("")
print ("percentage data that is missing :")
print (percentage_data_missing_round)
print ("")
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
print ("Knowing the amount of missing data both Street Number Suffix and ZipCode :")
missing_values_countx[['Street Number Suffix', 'Zipcode']]

print ("")
print ("knowing the percentage of both missing data :")
Street_number_suffix_missing_count=((missing_values_countx['Street Number Suffix'] / sf_permits.shape[0]) * 100)
Street_number_suffix_missing_count_round= round(Street_number_suffix_missing_count,2)
print ("")
print ("Percentage for missing Street Number suffix:")
print (Street_number_suffix_missing_count_round)
Zipcode_missing_count=((missing_values_countx['Zipcode'] / sf_permits.shape[0]) * 100)
Zipcode_missing_count_round= round(Zipcode_missing_count,2)
print ("")
print ("Percentage for missing ZipCode:")
print (Zipcode_missing_count_round)
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permitsx=sf_permits.copy()
sf_permits2=sf_permitsx.dropna()
print ("")
print ("Removing all the rows from sf_permits")
print (sf_permits2)
sf_permits2
# Now try removing all the columns with empty values. Now how much of your data is left?
print ("Removing all the columns with empty values , at least one missing value")
columns_with_na_dropped = sf_permits.dropna(axis=1)
print (columns_with_na_dropped.head())
columns_with_na_dropped.head()
print ("")
print ("Checking how much data did we lose")
print ("")
# just how much data did we lose?
print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
# Your turn! Try replacing all the NaN's in the sf_permit data with the one that
# comes directly after it and then 
subset_sf_permitsx = sf_permits.loc[:, 'Zipcode':'Street Number Suffix'].head()
subset_sf_permitsx
print (subset_sf_permitsx)

# replace all NA's with 0
subset_sf_permitsx.fillna(0)

# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
sf_permits1=subset_sf_permitsx.fillna(method = 'bfill', axis=0).fillna("0")
sf_permits1.head()
sf_permits2=subset_sf_permitsx.fillna(method = 'ffill', axis=0).fillna("0")
sf_permits2.head()
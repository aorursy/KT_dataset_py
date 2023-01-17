# modules we'll use
import pandas
import numpy

# read in all our data
nfl_data = pandas.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pandas.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
numpy.random.seed(0) 
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
nfl_data.sample(10)
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?

# your code goes here :)
# nfl_data.info()
sf_permits.sample(10)
# Looking at the sf_permits data

sf_permits.info()
# get the number of missing data points per column
missing_values_count_d1 = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count_d1
# how many total missing values do we have?
total_cells_d1 = numpy.product(nfl_data.shape)
total_missing_d1 = missing_values_count_d1.sum()

# percent of data that is missing
(total_missing_d1/total_cells_d1) * 100
# your turn! Find out what percent of the sf_permits dataset is missing

# Looking at the count of missing values
missing_values_count_d2 = sf_permits.isnull().sum()
missing_values_count_d2
# Getting the percentage of total data that is missing

total_cells_d2 = numpy.product(sf_permits.shape)
total_missing_d2 = missing_values_count_d2.sum()
(total_missing_d2 / total_cells_d2) * 100
# Looks like thier is huge amount of missing data
# look at the # of missing points in the first ten columns
missing_values_count_d1[0:10]
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped_d1 = nfl_data.dropna(axis=1)
columns_with_na_dropped_d1.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped_d1.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?

sf_permits.dropna()
# Now try removing all the columns with empty values. Now how much of your data is left?

columns_dropped_na_d2 = sf_permits.dropna(axis = 1)
columns_dropped_na_d2.head()
# Amount of data we lost

print("Columns in original data set: %d \n" % sf_permits.shape[1])
print("Columns with dropped NA's: %d" % columns_dropped_na_d2.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then 

# get a small subset of sf data
#sf_permits.info()
subset_sf_permits = sf_permits.iloc[10:20, 11:26].head(10)
subset_sf_permits
# Replace all NA's with zero

subset_sf_permits.fillna(0)
# Replacing all NA's with the upcoming value and the remaining with zeroes

subset_sf_permits.fillna(method = 'bfill', axis = 0).fillna("0")
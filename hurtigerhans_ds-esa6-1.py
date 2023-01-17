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



#look at the first 20 rows

sf_permits.sample(20)



#is data missing?

sf_permits.isnull()

#yes there is 





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



#amount of missing values

missing_value_count = sf_permits.isnull().sum().sum()



#amount of values

total_value_count = np.product(sf_permits.shape)



#missing values in percent

print(100 / total_value_count * missing_value_count, "% of the data is empty")

# look at the # of missing points in the first ten columns

missing_values_count[0:10]
look_sf = sf_permits[:100]

look_sf[["Zipcode","Street Number Suffix"]]



#Zipcode, if not existing wasn't recorded because every adress should have one.

#Street Number Suffix, if not existing is probably(!) not existing (but could also be forgotten but it's seen to be unlikely)
# remove all the rows that contain a missing value

nfl_data.dropna()
# remove all columns with at least one missing value

columns_with_na_dropped = nfl_data.dropna(axis=1)

columns_with_na_dropped.head(10)
# just how much data did we lose?

print("Columns in original dataset: %d \n" % nfl_data.shape[1])

print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?



#drop empty lines

empty_rows = sf_permits.dropna()

empty_rows



#There are 0 rows left
# Now try removing all the columns with empty values. Now how much of your data is left?

empty_col = sf_permits.dropna(axis=1)

empty_col



#There are 12 columns left

# get a small subset of the NFL dataset

subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()

subset_nfl_data
# replace all NA's with 0

subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 

# then replace all the reamining na's with 0

subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that comes directly after it and then replacing any remaining NaN's with 0

filled = sf_permits.fillna(method= 'bfill', axis=0).fillna(0)

filled.head(20)



#i don't understand the use of fillna(0) here. If the program is working as predicted there shoudln't be any empty values left after fillna(method=bfill'...)
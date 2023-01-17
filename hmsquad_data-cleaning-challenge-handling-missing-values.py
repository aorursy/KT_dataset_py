# modules we'll use

import pandas as pd

import numpy as np



# read in all our data

sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")



# set seed for reproducibility

np.random.seed(0) 
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?

sf_permits.sample(5)

# your code goes here :)
 

#Find out what percent of the sf_permits dataset is missing

# get the number of missing data points per column

missing_values_count = sf_permits.isnull().sum()



# look at the # of missing points in the first ten columns

missing_values_count



# how many total missing values do we have?

total_cells = np.product(sf_permits.shape)

total_missing = missing_values_count.sum()



# percent of data that is missing

(total_missing/total_cells) * 100
# look at the # of missing points in the first ten columns

missing_values_count[0:10]
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?

# remove all columns with at least one missing value

columns_with_na_dropped = sf_permits.dropna(axis=1)

columns_with_na_dropped.head()
# Now try removing all the columns with empty values. Now how much of your data is left?

# just how much data did we lose?

print("Columns in original dataset: %d \n" % sf_permits.shape[1])

print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# replacing all the NaN's in the sf_permits data with the one that

# comes directly after it and then replacing any remaining NaN's with 0

sf_permits.fillna(method = 'bfill', axis=0).fillna(0)
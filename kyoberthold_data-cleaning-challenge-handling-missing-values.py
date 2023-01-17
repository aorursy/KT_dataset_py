# modules we'll use

import pandas as pd

import numpy as np



# read in all our data

data = pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")



# set seed for reproducibility

#np.random.seed(0) 
# look at a few rows of the nfl_data file. I can see a handful of missing data already!

data.sample(5)
# your turn! Look at a couple of rows from the dataset. Do you notice any missing data?



data.Population[2800]
# get the number of missing data points per column

missing_values_count = data.isnull().sum()



# look at the # of missing points in the first ten columns

missing_values_count[0:10]
# how many total missing values do we have?

total_cells = np.product(data.shape)

total_missing = missing_values_count.sum()



# percent of data that is missing

(total_missing/total_cells) * 100
# look at the # of missing points in the first ten columns

missing_values_count[0:10]
# remove all the rows that contain a missing value

data.dropna()
# remove all columns with at least one missing value

columns_with_na_dropped = data.dropna(axis=1)

columns_with_na_dropped.head()
# just how much data did we lose?

print("Columns in original dataset: %d \n" % data.shape[1])

print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
subset_nfl_data = data.loc[:, 'Status':'Schooling'].head()

subset_nfl_data
# replace all NA's with 0

subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 

# then replace all the reamining na's with 0

subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# modules we'll use

import pandas as pd

import numpy as np



# read in all our data

nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")

sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")



# set seed for reproducibility

np.random.seed(0) 
# look at a few rows of the nfl_data file. I can see a handful of missing data already!

#This method returns a random sample of the data from our dataframe

nfl_data.sample(5)
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?



# your code goes here :)



#Use the same sample function used above.Lots of Columns have NaN's based on initial overview.



sf_permits.sample(10)
# get the number of missing data points per column

missing_values_count = nfl_data.isnull().sum()



# look at the # of missing points in the first ten columns

missing_values_count[0:10]
'''how many total missing values do we have.

Gets the value of all the values in our file(cells) and gets the total count of null values(cells)'''



total_cells = np.product(nfl_data.shape)

total_missing = missing_values_count.sum()



# percent of data that is missing

(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing.Around 26% of the Data is Missing

total_cells=sf_permits.shape[0]*sf_permits.shape[1]



'''Empty values and total values calculated separately'''

total_null=sf_permits.isnull().sum().sum()



total_percent_missing=(total_null/total_cells)*100



total_percent_missing
# look at the # of missing points in the first ten columns

#Imputation is the process of filling empty values in the dataset

missing_values_count[:]
sf_permits.columns

'''Around 95% of values for the Street Number Suffix are empty.So it is highly likely that for many streets

in SF,there is no Suffix for the Street Numbers while everyplace in the world has its own zipcode(mostly) so

it is likely that these numbers were not recorded/misplaced'''



temp_col=sf_permits['Street Number Suffix']

zipCode=sf_permits['Zipcode']



print(temp_col.isnull().sum())

print(temp_col.shape)

print(zipCode.isnull().sum())
# remove all the rows that contain a missing value

nfl_data.dropna()
# remove all columns with at least one missing value

columns_with_na_dropped = nfl_data.dropna(axis=1)

columns_with_na_dropped.head()
# just how much data did we lose?

print("Columns in original dataset: %d \n" % nfl_data.shape[1])

print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?

'''All the rows have atleast one empty value so all the data is cleared.

Better is to put a condition in dropna for removal of rows'''



print('OG shape is',sf_permits.shape)



sf_permits=sf_permits.dropna()



print('After Deletion Shape is',sf_permits.shape)
# Now try removing all the columns with empty values. Now how much of your data is left?

#All rows are presend but around 75% of the columns have been deleted

print('OG shape is',sf_permits.shape)



sf_permits=sf_permits.dropna(axis=1)



print('After Deletion Shape is',sf_permits.shape)
# get a small subset of the NFL dataset(All Rows and columns from EPA to the Season Column)

subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()

subset_nfl_data
# replace all NA's with 0

subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 

# then replace all the reamining na's with 0

subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that

# comes directly after it and then replacing any remaining NaN's with 0



sf_permits_copy=sf_permits[:]



'''Allows you to fill previous or next valid value.Useful for time series where there is a logical order between

data'''

sf_permits_copy.fillna(method='bfill',axis=0).fillna(0)
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



sf_permits.sample(10)
#nan = nfl_data[nfl_data.isna().any(axis=1)]

def find_nan(df):

    return (df[df.isnull().any(axis=1)])



find_nan(nfl_data).head()

# nfl_data.isnull().sum()

def misng_na_Ser_val_count(df):

    misng_na_Ser_val_count = df.isnull().sum()

    return(misng_na_Ser_val_count[:])



misng_na_Ser_val_count(nfl_data)
misng_na_Ser_val_count(sf_permits).sample(10)
# get the number of missing data points per column

missing_values_count = nfl_data.isnull().sum()



# look at the # of missing points in the first ten columns

missing_values_count[0:5]

# nfl_data.columns
def percentage_missing(df):

    perc = ((df.isnull().sum()).sum()/np.product(df.shape))*100

    return(perc)

print('Total null cell in the dataframe is: ',nfl_data.isnull().sum().sum())

print('Total cells in the dataframe is: ',np.product(nfl_data.shape))

print('Percentage missing is: ',percentage_missing(nfl_data))
def percentage_missing(df):

    perc = ((df.isnull().sum()).sum()/np.product(df.shape))*100

    return(perc)

print('Total null cell in the dataframe is: ',sf_permits.isnull().sum().sum())

print('Total cells in the dataframe is: ',np.product(sf_permits.shape))

print('Percentage missing is: ',percentage_missing(sf_permits))
# how many total missing values do we have?

total_cells = np.product(nfl_data.shape)

print(total_cells)

total_missing = missing_values_count.sum()

print(total_missing)

# percent of data that is missing

print((total_missing/total_cells) * 100)
# your turn! Find out what percent of the sf_permits dataset is missing



tot_nul_sum = sf_permits.isnull().sum().sum()

X = np.product(sf_permits.shape)

print('Percent of the sf_permits dataset missing is: ',(tot_nul_sum/X)*100)



# print("Total NaN in Dataframe" , nfl_data.isnull().sum().sum(), sep='\n')





# percentage_missing(sf_permits)
# TimesSec column total missing values

nfl_data['TimeSecs'].isnull().sum()
# look at the # of missing points in the first ten columns

missing_values_count[0:10]
nfl_data.loc[nfl_data.down == 3]
sf_permits.loc[:, ['Street Number Suffix', 'Zipcode']].dropna(axis=0).head()
sf_permits.loc[:, ['Street Number Suffix', 'Zipcode']].head()
# remove all the rows that contain a missing value

nfl_data.dropna()
# remove all columns with at least one missing value

columns_with_na_dropped = nfl_data.dropna(axis=1)

columns_with_na_dropped.head()
columns_with_na_dropped.shape[0]
# just how much data did we lose?

print("Columns in original dataset: %d \n" % nfl_data.shape[1])

print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?

sf_permits.dropna(axis=0)
# Now try removing all the columns with empty values. Now how much of your data is left?

sf_new = sf_permits.dropna(axis=1)
print('without dropped columns: %d' %sf_permits.shape[1])

print('with columns dropped: %d' %sf_new.shape[1])
# get a small subset of the NFL dataset

subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head(10)

subset_nfl_data
# replace all NA's with 0

subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 

# then replace all the reamining na's with 0

subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# sf_permits.isnull().head(10)

sf_permits.head()
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that

# comes directly after it and then replacing any remaining NaN's with 0

# sf_permits.isnull().sum()

new_df = sf_permits.fillna(method='bfill', axis=0).fillna(0).sample(10)

new_df
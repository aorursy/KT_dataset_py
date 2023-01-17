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
sf_permits.head(5)
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
sf_missing_per_column = sf_permits.isnull().sum()
sf_total_cells = np.product(sf_permits.shape)
sf_total_missing_cells = sf_missing_per_column.sum()

(sf_total_missing_cells / sf_total_cells ) * 100
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
#sf_permits[['Street Number Suffix', 'Zipcode']].isnull()
print('Street No Suffix. Total records: %d' % sf_permits.shape[0])
street_no_suffix_missing = sf_permits[sf_permits['Street Number Suffix'].isnull()].shape[0]
print('Total N/A %d %.3f' % (street_no_suffix_missing, street_no_suffix_missing / sf_permits.shape[0]))
sf_permits[sf_permits['Street Number Suffix'].isnull()].sample(5) # looks that they don't exist
zipcode_missing = sf_permits[sf_permits['Zipcode'].isnull()].shape[0]
print('Total N/A %d %.3f' % (zipcode_missing, zipcode_missing / sf_permits.shape[0]))
sf_permits[sf_permits['Zipcode'].isnull()].sample(10) # looks that they weren't recorded
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_dropped_na = sf_permits.dropna()
sf_dropped_na.shape
# Now try removing all the columns with empty values. Now how much of your data is left?
sf_col_dropped_na = sf_permits.dropna(axis=1)
print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % sf_col_dropped_na.shape[1])
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
cols_with_na = sf_permits.columns.drop(sf_col_dropped_na.columns)
cols_with_na
# sf_permits_na = sf_permits[cols_with_na]
# My strategy: Analyse chunks of columns that seems to be related
# sf_permits_na = sf_permits.loc[:,'Permit Number':'Unit Suffix']
# sf_permits_na = sf_permits.loc[:,'Description':'Number of Proposed Stories']
# sf_permits_na = sf_permits.loc[:,'Voluntary Soft-Story Retrofit':'TIDF Compliance']
sf_permits_na = sf_permits.loc[:,'Existing Construction Type':'Record ID']
sf_permits_na.head()
column = 'Zipcode'
# sf_permits_na[sf_permits_na[column].isnull()] # to analyse the N/A records
sf_permits_na[sf_permits_na[column].isnull()] # to analyse non N/A records
# sf_permits[['Proposed Use', column]][sf_permits_na[column].isnull()]

sf_permits_na[column].value_counts()
df_with_imputation = sf_permits.copy()
df_with_imputation.shape
df_with_imputation['Street Number Suffix'].fillna('', inplace=True)
df_with_imputation['Street Suffix'].fillna('', inplace=True)
df_with_imputation['Unit'].fillna(0, inplace=True)
df_with_imputation['Unit Suffix'].fillna('', inplace=True)
df_with_imputation['Description'].fillna('', inplace=True)
df_with_imputation['Issued Date'].fillna(df_with_imputation['Filed Date'], inplace=True)
df_with_imputation['First Construction Document Date'].fillna(df_with_imputation['Filed Date'], inplace=True)
df_with_imputation['Structural Notification'].fillna('N', inplace=True)
df_with_imputation['Number of Existing Stories'].fillna(0.0, inplace=True)
df_with_imputation['Number of Proposed Stories'].fillna(0.0, inplace=True)
df_with_imputation['Voluntary Soft-Story Retrofit'].fillna('N', inplace=True)
df_with_imputation['Fire Only Permit'].fillna('N', inplace=True)
df_with_imputation['Estimated Cost'].fillna(0.0, inplace=True)
df_with_imputation['Revised Cost'].fillna(0.0, inplace=True)
df_with_imputation['Existing Units'].fillna(0.0, inplace=True)
df_with_imputation['Site Permit'].fillna('N', inplace=True)
df_missing_per_column = df_with_imputation.isnull().sum()
df_total_cells = np.product(df_with_imputation.shape)
df_total_missing_cells = df_missing_per_column.sum()

df_missing = (df_total_missing_cells / df_total_cells ) * 100
sf_missing = (sf_total_missing_cells / sf_total_cells ) * 100

print(f'Dataset improvement in terms of missing values\n % mising values Before: {sf_missing}\n % missing values Now: {df_missing}')
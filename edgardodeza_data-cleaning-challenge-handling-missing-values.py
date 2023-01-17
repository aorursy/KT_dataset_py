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
# your code goes here :)

sf_permits.sample(5)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# determine the number of missing values in each column
# We store these values in an array called 'num_missing_values_per_column'
# See: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.isnull.html
num_missing_values_per_column = sf_permits.isnull().sum()

# Missing numbers per column
print(num_missing_values_per_column)

# total number of missing values
total_missing = num_missing_values_per_column.sum()
print("\nThe number of total missing values is {}".format(total_missing))
# We determine the percentage of missing values

# First we determine the number of total entries in our data frame
# sf_permits.shape returns a tuple = (rows, columns) that we can unpack
rows, columns = sf_permits.shape
total_number_entries = rows * columns

# The percentage of missing values is calculated as the ratio of
# number of missing values / total number of entries
percentage_missing = total_missing / total_number_entries * 100

print("The 'sf_permits' dataframe has {} rows and {} columns.".format(rows, columns))
print("This means we have {} entries in our dataframe.".format(total_number_entries))

print("\nIn total {} values are missing.".format(total_missing))

print("\nThis means {}% values are missing.".format(round(percentage_missing, 1)))
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# Let's first have a look at the column 'Street Number Suffix'
street_number_suffix = sf_permits["Street Number Suffix"]

# The number of NaN values in this column is:
print("The number of NaN values in the 'Street Number Suffix' columns is: ", street_number_suffix.isnull().sum())

# calculate percentage of missing values in that column
print("Our 'sf_permits' dataframe has {} rows.".format(sf_permits.shape[0]))
print("\nThis means {}% of the values in this column are missing.".format(round(street_number_suffix.isnull().sum() / sf_permits.shape[0] * 100,1)))

print("\nIt is highly unlikely that so many values are missing because they were not recorded.")
print("Instead we can assume that almost 99% of the streets do not have a Street Number Suffix.")
# Let's have a look at the Zipcode column
zipcode = sf_permits["Zipcode"]
num_missing_values_in_zipcode = zipcode.isnull().sum()
print("Number of missing values in the column 'Zipcode': ", num_missing_values_in_zipcode)
print("This means {}% in the column 'Zipcode' are missing.".format(num_missing_values_in_zipcode / sf_permits.shape[0] * 100))

print("\nSo, about 1% of the Zipcode values are missing. We can assume that they simply have not been recorded.")
print("Every building permit in San Francisco should have a zipcode.")
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
sf_permits_all_rows_with_na_removed = sf_permits.dropna()
print("The size of our new dataframe after removing all NaN values is: ", sf_permits_all_rows_with_na_removed.shape)
# Now try removing all the columns with empty values. Now how much of your data is left?
sf_permits_all_columns_with_na_removed = sf_permits.dropna(axis=1)
print("The size of our new dataframe after removing all columns with NaN is: ", sf_permits_all_columns_with_na_removed.shape)

# Compare the number of columns before and after
print("\nnumber of columns in original dataframe: ", sf_permits.shape[1])
print("\nnumber of columns in dataframe with dropped NaN columns: ", sf_permits_all_columns_with_na_removed.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
# Let's first create a subset of the sf_permits dataframe
subset_sf_permits = sf_permits.loc[:, "Existing Construction Type":"Proposed Construction Type Description"].head(10)
subset_sf_permits
# Now let's replace every NaN value by -1
subset_sf_permits.fillna(-1)
# replace all NA's with the value that comes directly after it in the same column, 
# then replace all the reamining na's with -1
subset_sf_permits.fillna(method = 'bfill', axis=0).fillna(-1)
# We can also use the previous value from above to fill replace a NaN value
subset_sf_permits.fillna(method = 'ffill', axis=0).fillna(-1)
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
#sf_permits.sample(5)
#sf_permits.isnull()
#-----List rows with NaN in data frame
sf_permits[sf_permits[0:0].isnull()]
#sf_permits.shape

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
Missing_Data_Count = sf_permits.isnull().sum()
#Missing_Data_Count
#Missing_Data_Count.head(10)
Total_Cells_Data = np.product(sf_permits.shape)
Total_Missing_Data = Missing_Data_Count.sum()
(Total_Missing_Data/Total_Cells_Data)*100

# around 26% of data has missing values

# look at the # of missing points in the first ten columns
missing_values_count.sort_values(ascending=False).head(20)
sf_permits.shape
# Looking at the sf_permits Data set
#sf_permits.isnull().sum().sort_values(ascending=False)
#sf_permits.describe()
#sf_permits["Street Number Suffix"].describe()
#sf_permits["Zipcode"].describe()
sf_permits["Street Number Suffix"].notnull().sum()
#2216
sf_permits["Street Number Suffix"].isnull().sum()
#196684
#Ratio of missing data
(196684/198900)*100
#98.8% of rows have missing data
sf_permits["Street Number Suffix"].describe()
#count     2216
#unique      18
#top          A
#freq      1501
#Name: Street Number Suffix, dtype: object

# Drop the column

#pd.options.display.max_columns = None
#display(sf_permits).sample

sf_permits[sf_permits["Street Number Suffix"].notnull()]
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
# Now try removing all the columns with empty values. Now how much of your data is left?
Col_Na_Droped = sf_permits.dropna(axis = 1)
Col_Na_Droped.head()

print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % Col_Na_Droped.shape[1])
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
Replace_Temp = sf_permits.drop(columns=["Street Number Suffix"])
#Replace_Temp
Replaced_Na = Replace_Temp.fillna(method = 'bfill', axis=0).fillna("0")
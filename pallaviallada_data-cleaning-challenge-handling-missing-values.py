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
sf_permits.head()
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
null_values_columns = sf_permits.isnull().sum()
total_number_of_empty_cells = null_values_columns.sum()

total_number_of_cells = np.product(sf_permits.shape)
percent_of_empty_cells = (total_number_of_empty_cells/total_number_of_cells)*100
percent_of_empty_cells
null_values_columns.sort_values(ascending=False)
from matplotlib import pyplot as plt
import numpy as np
# Get current size
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=20 # Increases the figure width for a neat look
plt.rcParams["figure.figsize"] = fig_size
x,y = null_values_columns.index.tolist(), null_values_columns.values
x_pos = np.arange(len(x)) # Convert the labels to indexes as bar() method takes only scalars to plot
plt.bar(x_pos, y, align='center', alpha=0.25)
plt.xticks(x_pos,x, rotation=90) # Map the indexes used for the x axis to the labels for readability 
plt.show()
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# Let's identify the rows with empty Zipcode. Zipcode is an integral part of any address. 
# Also, we should check if there are any empty Street Number, Street Name at all.
print("Empty Street Number...", sf_permits['Street Number'].isnull().sum())
print("Empty Street Name...", sf_permits['Street Name'].isnull().sum())
print("Empty Street Suffix...", sf_permits['Street Suffix'].isnull().sum())
print("Empty Zipcode...", sf_permits['Zipcode'].isnull().sum())
# Street Number and Name have alsways been provided, even when there is no Zipcode, so having empty values for Zipcode is missing data
# which can be filled by looking at other same Street data. We will ignore Street Suffix for now.

null_zips = sf_permits.loc[(sf_permits.Zipcode.isnull()),['Street Number','Street Name']]
street_detail_tuple = list(zip(*[null_zips[i].values.tolist() for i in null_zips]))
print(type(street_detail_tuple))

# Create a dictionary of zipcode where the tuple (street number, street name) is the key and zipcode is its value.
# This is done using records which have a not null zipcode. We will use this as our master and fill the null zipcode records,
# if the street number and name match.
dict1 = {}
for index,row in sf_permits[sf_permits.Zipcode.notnull()].iterrows():
    num, nam, zipc = row['Street Number'], row['Street Name'], row['Zipcode'] 
    dict1[(num,nam)] = zipc

print(sf_permits.loc[452])
print(sf_permits.loc[577])
# Lookup the zipcode in the dictionary for which we have street number and name, but the zipcode is not filled in.
for index,row in sf_permits[sf_permits.Zipcode.isnull()].iterrows():
    tup = (row['Street Number'],row['Street Name'])
    valu = dict1.get(tup)
    if(valu is not None):
        sf_permits.loc[index,'Zipcode'] = valu
print(sf_permits.loc[452])
print(sf_permits.loc[577])

# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
print("Total rows before cleanup...", sf_permits.shape[0])
#print("Number of rows with atleast one NaN...", sf_permits.isnull().sum())
sf_permits_cleaned = sf_permits.dropna()
print("Total rows after cleanup...", sf_permits_cleaned.shape[0])
# Now try removing all the columns with empty values. Now how much of your data is left?
print("Total columns before cleanup...", sf_permits.shape[1])
sf_permits_cleaned_cols = sf_permits.dropna(axis=1)
print("Total columns after cleanup...", sf_permits_cleaned_cols.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing any remaining NaN's with 0
sf_permits_no_null = sf_permits.fillna(method='bfill',axis=0).fillna(0)
print("Number of NaN cells...", sf_permits_no_null.isnull().sum().sum())
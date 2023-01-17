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
sf_permits.sample(5)
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
missing_values_in_sf = sf_permits.isnull().sum()
(missing_values_in_sf.sum()/np.product(sf_permits.shape))*100

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
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
# Oh no data left! how suprising!
# Now try removing all the columns with empty values. Now how much of your data is left?
drop_columns_sf=sf_permits.dropna(axis=1)

print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % drop_columns_sf.shape[1])
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

"""well I will follow the way I defined above. before imputing, i will first try to see what 
values are actually missing. here my threshold will be 50%"""
missing_percentages = (missing_values_in_sf / sf_permits.shape[0])*100

sf_permits_toModify = sf_permits.copy()

for colname,perc in enumerate(missing_percentages):
    if perc > 50:
        print(missing_percentages.index[colname])
        #remove those columns one by one by the following
        sf_permits_toModify.drop(missing_percentages.index[colname],axis=1, inplace=True)
#this will tell me, or guide me for the columns whose NaN actually do not exist rather then missing. 
print("I do not see a reason to keep these columns in my dataframe after checking what they are")


""" 

before imputing, I need to see which columns are numbers and strings as I might use a "mean" or a "median" method to
fill the NaNs. for this purpose I will check the type of the columns one by one

"""

for i in sf_permits_toModify.columns:
    if sf_permits_toModify[i].dtype != "object":
        print(i, sf_permits_toModify[i].dtype,"\n")

#Starting from the first numeric column, I want to understand whether there is a dominance pattern or equally shared 
from collections import Counter
Counter(sf_permits_toModify.iloc[:,1])
# Obviously, the Permit Type column is not made of continuous values. So this could actually be something that we
#could later on predict based on our other values. I will leave this columns like this for now.

""" 

When I look at the Number of Stories existing and proposed, they match in a very high percentage.

"""
stories = sf_permits_toModify.iloc[:,15:17]
noNas_stories = stories.dropna()
C = np.where(noNas_stories.iloc[:,0] == noNas_stories.iloc[:,1],"yes","no")
yes_no=Counter(C)
print("the percentage of perfect match is: ",(yes_no["yes"]/(yes_no["yes"]+yes_no["no"]))*100)

""" 

So, I believe merging both the columns and keeping the merged columns instead of the two would be more
beneficial.

"""

merged = stories.iloc[:,0].fillna(stories.iloc[:,1])




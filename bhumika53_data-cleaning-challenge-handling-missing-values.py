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
missing_values_count1=sf_permits.isnull().sum()
#it will calculate the sum for column by column and display it in the form of a matrix. 
missing_values_count1

#sum of all missing values
missing_values_count1.sum()
#See % of null data for each column. 
sf_permits.shape #198900 rows and 43 columns 
percentage_null=[]
percentage_null=[sf_permits[column].isnull().sum()/sf_permits.shape[0] for column in sf_permits.columns]
#print(column,percentage_null)
#have to convert the percentage_null list into an array and then to convert it into dataframe
percentage_null=np.array(percentage_null)
#we can also make a dictionary out of it here. 
d = {'Column': sf_permits.columns, 'Percentage_null': percentage_null}
df = pd.DataFrame(data=d)
df
#Here, we want to know which columns have more than 50% of their values as null. 
filter1= df.Percentage_null>0.50
df[filter1]
df
#to find out more about the existing values of the Street Number suffix. 
filter2=sf_permits['Street Number Suffix'].notnull()==True
sf_permits[filter2].sample(10)
#to check the false values 
filter3=sf_permits['Street Number Suffix'].notnull()==False
sf_permits[filter3].sample(10)

#there is no correlation between the year and the SNS. 


#to know the entire ppercentage of the missing values inside the entire dataset. 
total=np.product(sf_permits.shape)
total_percentage_null=missing_values_count1.sum()/total*100
total_percentage_null
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
nfl_data.dropna()
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
#axis =1 , is to drop the columns with na value. 
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits.dropna()
# Now try removing all the columns with empty values. Now how much of your data is left?
sf_permits.dropna(axis=1).shape
sf_permits.shape
print("earlier size:", sf_permits.dropna(axis=1).shape)
print('new size:',sf_permits.shape)
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
sf_permits.fillna(method='bfill',axis=0).fillna(0)
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
print("sf_permits columns name with null value in the column")
sf_permits_missing_columns = list(sf_permits.iloc[0][sf_permits.isnull().any()].index)
print(sf_permits_missing_columns)
sf_permits.sample(5)
# your code goes here :)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize = (16,5))
missing_columns = list(missing_values_count[missing_values_count != 0].index)
sns.heatmap(nfl_data[missing_columns].isnull(),yticklabels=False,cbar=False,cmap='viridis')
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing
total_cells = np.product(sf_permits.shape)
total_missing =sf_permits.isnull().sum().sum()

print((total_missing/total_cells) * 100,"%")
plt.figure(figsize = (16,5))
sns.heatmap(sf_permits[sf_permits_missing_columns].isnull(),yticklabels=False,cbar=False,cmap='viridis');
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
plt.figure(figsize = (4,2))
sns.heatmap(sf_permits[["Street Number Suffix","Zipcode"]].isnull(),yticklabels=False,cbar=False,cmap='viridis');
sf_permits[sf_permits["Street Number Suffix"].isnull() == False ].sample(3)
# probably they just don't have street number suffix
# even if some value are accidently not recorded there is not much information for us to fill the missing value by the dataset
# in this case, we would like to match the data set with some open resource such as OpenStreetMap or GoogleMap 
sf_permits[sf_permits["Zipcode"].isnull() == True ].sample(3)
# They are probably not recorded but should have value
# in this case we might check same block, lot, street 
# to get the zipcode since there is no missing values in these columns !if exist!!
sf_permits.groupby(['Block','Lot']).Zipcode.describe()
# remove all the rows that contain a missing value
nfl_data.dropna() 
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
### !!!!! None != NaN !!!!
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits_clean = sf_permits.dropna(axis = 0)
print("Columns in original dataset: %d \n" % sf_permits.shape[0])
print("Columns with na's dropped: %d" % sf_permits_clean.shape[0])
# Now try removing all the columns with empty values. Now how much of your data is left?
sf_permits_clean = sf_permits.dropna(axis = 1)
print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % sf_permits_clean.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
### method : {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}, default None
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
list(sf_permits.iloc[0][sf_permits.isnull().any() == False].index)
# same block, lot has same zipcode
(sf_permits.groupby(['Block','Lot']).Zipcode.max().dropna() == sf_permits.groupby(['Block','Lot']).Zipcode.min().dropna()).all()
blocklothelper = sf_permits.groupby(['Block','Lot']).Zipcode.max()
blocklothelper.head()#.loc['0010','001']
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then 
# try filling zipcode
def impute_zip(cols):
    b, l, z = cols
    if pd.isnull(z):
        if pd.notnull(blocklothelper.loc[b,l]):
            return blocklothelper.loc[b,l]
        else:
            return z
    else:
        return z
print('missing value in zipcode before imputation:', sf_permits.isnull().Zipcode.sum())
print('missing value in zipcode after imputation:', sf_permits[['Block','Lot','Zipcode']].apply(impute_zip,axis = 1).isnull().sum())
sf_permits[sf_permits_missing_columns].info()
### fill float type by mode
print('missing total value before imputation:', sf_permits.isnull().sum().sum())
for col_name in sf_permits_missing_columns:
    if sf_permits[col_name].dtype == 'float64':
        sf_permits[col_name] = sf_permits[col_name].value_counts().sort_index(ascending = False).iloc[0]
print('missing total value after imputation:', sf_permits.isnull().sum().sum())
sf_permits.head()
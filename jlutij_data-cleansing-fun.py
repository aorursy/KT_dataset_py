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
print ("use sf_permits.info() to get permits metadata types \n")
print ("use sf_permits.describe() to get statistics of permits data")
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
# your turn! Find out what percent of the sf_permits dataset is missing
permits_missing_Values_Count = sf_permits.isnull().sum()
##print (permits_missing_Values_Count)
total_permits = np.product(sf_permits.shape)
permits_missing = permits_missing_Values_Count.sum()
percent_permits_missing = (permits_missing/total_permits)*100
print('Percent.of Permits missing values %f' % percent_permits_missing )
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
print("Validating missing values in Street Number Suffic and ZipCode")
sf_permits['Zipcode'].describe()
sf_permits[(sf_permits['Street Number Suffix'].isnull()== True) & (sf_permits['Zipcode'].isnull()== True)].head(5)
sf_permits[(sf_permits['Street Number Suffix'].isnull()== False) & (sf_permits['Zipcode'].isnull()== True)].head(5)
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_Copy = sf_permits.copy()
sf_Copy.dropna(how='all')
print("Rows in original dataset: %d \n" % sf_permits.shape[0] )
print("Rows in new dataset with na's dropped: %d" % sf_Copy.shape[0])
# Now try removing all the columns with empty values. Now how much of your data is left?
permits_na_dropped = sf_permits.dropna(axis=1)
permits_na_dropped.head()
print("Columns in original dataset: %d \n" % sf_permits.shape[1] )
print("Columns in new dataset with na's dropped: %d" % permits_na_dropped.shape[1])
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
sf_permits.fillna(method='bfill', axis=0).fillna("0")
import pandas as pd
print ("Reading SanFranscisco open dataset")
sf_address = pd.read_csv("../input/openaddress-sanfrancisco/san_francisco.csv", low_memory=False)
sf_Copy = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")
## drop empty rows
sf_Copy.dropna()
sf_Copy['Street Name'] = sf_Copy['Street Name'].apply(lambda x: x.upper())
sf_Copy['Street Number'] = sf_Copy['Street Number'].apply(str)
sf_address['STREET'] = sf_address['STREET'].apply(str)
new_df = pd.merge(sf_Copy, sf_address,  how='left', left_on=['Street Number','Street Name'], right_on = ['NUMBER','STREET'])

sf_address[(sf_address['STREET'] == 'BUCHANAN') & (sf_address['NUMBER'] == '3111') ].head(5)
new_df[new_df['Street Name'] == 'BUCHANAN'].head(5)
new_df.loc[new_df['Zipcode'].isnull(),'Zipcode'] = new_df['POSTCODE']
new_df.head(5)
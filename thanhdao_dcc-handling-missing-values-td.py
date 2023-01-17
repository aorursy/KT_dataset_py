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
# sf_permits.sample(10)
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

# your turn! Find out what percent of the sf_permit dataset is missing
missing_values_count1= sf_permits.isnull().sum()
total_cells1 = np.product(sf_permits.shape)
total_missing1 = missing_values_count1.sum()
(total_missing1 / total_cells1) * 100
missing_values_count1
# look at the # of missing points in the first ten columns
missing_values_count
missing_values_count1
#sf_permits.sample(5)
sf_permits.iloc[:,8:10]
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
sf_permits_colred = sf_permits.dropna(axis=1)
print(sf_permits_colred.shape[1])
print(sf_permits.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
# Your turn! Try replacing all the NaN's in the sf_permit data with the one that
# comes directly after it and then 
subset1 = sf_permits.iloc[:,1:15]
subset1.head(20)
subset1.fillna(method='bfill', axis=0).fillna('0')
#add new data source OpenAddresses U.S West & 
# rerun all the queries because all the codes were somehow gone after loading new data
# find the state that I want to import from this dataset - it is Sanfrancisco -> CA california
#sf_permits.Location
# read data into df 
ca_data = pd.read_csv("../input/openaddresses-us-west/ca.csv")
ca_data.head()

#find ways to map this two dataset together: either Location OR street name. Location has less data but city name can be duplicate across state
sf_permits.columns
#sf_permits.iloc[:,38:43]
#sf_permits[sf_permits.Zipcode.isnull()]
clean_permits = sf_permits.loc[:,['Block', 'Lot', 'Street Number', 'Street Number Suffix', 'Street Name', 
                            'Street Suffix','Neighborhoods - Analysis Boundaries', 'Zipcode','Location']]
clean_permits['STREET'] = clean_permits['Street Name'].astype(str).str.upper() + ' '+ clean_permits['Street Suffix'].astype(str).str.upper()
#clean_permits= clean_permits.set_index('STREET')
clean_permits.head()
#result = pd.merge(clean_permits, ca_data, how='left', on=['STREET'])
#new_data = clean_permits.loc[:,['STREET','Street Number Suffix']].set_index('STREET')
#new_data.update(ca_data.set_index('STREET'))
#df['Street Number Suffix'] = new_data.values
#new_data.head(10)

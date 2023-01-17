!ls -ltr ../input
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
sf_permits.sample(10)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
missing_values_count.sort_values(ascending=False)[0:10]
a1 = np.array([2, 4, 6])
print(f"the product of all array elements: {np.product(a1)}")
a2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"shape of array a2: {a2.shape}")
print(f"the product of the diemensions of the 2-d array a2: {np.product(a2.shape)}")
print("this is the total number of elements or cells in this 2-d array")
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
nfl_missing_percent = (total_missing/total_cells) * 100
print(f"nfl_missing_percent: {nfl_missing_percent:.2f}%")
# your turn! Find out what percent of the sf_permits dataset is missing

# get the number of missing data points per column
sf_missing_values_count = sf_permits.isnull().sum()
#sf_missing_values_count

# look at the # of missing points in the first ten columns
sf_missing_values_count[0:10]

# total number of cells in sf permits dataset
sf_total_cells = np.product(sf_permits.shape)

# total number of cells with missing values in sf permits dataset
sf_total_missing = sf_missing_values_count.sum()

# percent of data missing in sf permits dataset
sf_missing_percent = (sf_total_missing / sf_total_cells) * 100
print(f"sf_missing_percent: {sf_missing_percent:.2f}%")
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
sf_missing_values_count[['Street Number Suffix', 'Zipcode']]
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits_dropped_rows_with_nans = sf_permits.dropna()
sf_permits_dropped_rows_with_nans
# Now try removing all the columns with empty values. Now how much of your data is left?
sf_permits_dropped_cols_with_nans = sf_permits.dropna(axis=1)
print(f"Number of columns in original dataset: {sf_permits.shape[1]}")
print(f"Number of columns after dropping colums with nas: {sf_permits_dropped_cols_with_nans.shape[1]}")
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
sf_permits.fillna(method="bfill", axis=0).fillna(0)
oa_sf_df = pd.read_csv('../input/openaddress-san-francisco/san_francisco/us/ca/san_francisco.csv')
oa_sf_df.sample(5)
# count of null values by column
oa_missing_values_count = oa_sf_df.isnull().sum()
permits_missing_values_count = sf_permits.isnull().sum()

print(f"OpenAddress number of rows with null longitude: {oa_missing_values_count['LON']}")
print(f"OpenAddress number of rows with null latitude: {oa_missing_values_count['LAT']}")
print(f"OpenAddress number of rows with null street number: {oa_missing_values_count['NUMBER']}")
print(f"OpenAddress number of rows with null street name: {oa_missing_values_count['STREET']}")
print("*" * 3)
# Location contains a tuple longitude and latitude values
print(f"sfpermits number of rows with null location: {permits_missing_values_count['Location']}")
print(f"sfpermits number of rows with null street number: {permits_missing_values_count['Street Number']}")
print(f"sfpermits number of rows with null street name: {permits_missing_values_count['Street Name']}")
print(f"sfpermits number of rows with null street name suffix: {permits_missing_values_count['Street Suffix']}")
oa_sf_df[oa_sf_df["NUMBER"].str.contains('A')].sample(5)
streets = pd.Series(['BUSH ST', 'BUSH ST', 'SUTTER ST', 'PACIFIC AVE'])
numbers = pd.Series(['100', '200', '60', '80'])
zipcodes = pd.Series(['4339320', np.NaN, np.NaN, np.NaN])
a = {'Street': streets, 'Number': numbers, 'Zipcode': zipcodes}
a_df = pd.DataFrame.from_dict(a)
a_df['Street-Number'] = a_df['Street'] + '-' + a_df['Number']

streets2 = pd.Series(['CLAY ST', 'BUSH ST', 'BERRY ST', 'SUTTER ST', 'PACIFIC AVE'])
numbers2 = pd.Series(['20', '100', '40', '60', '80'])
zipcodes2 = pd.Series(['40549', '4339320', '40545', '60213', '12345'])
irrelevant = pd.Series(['A', 'B', 'C', 'D', 'E'])
b = {'Street': streets2, 'Number': numbers2, 'Zipcode': zipcodes2, 'Superfluous': irrelevant}
b_df = pd.DataFrame.from_dict(b)
b_df['Street-Number'] = b_df['Street'] + '-' + b_df['Number']
print(a_df)
print("\n")
print(b_df)
a_df['Zipcode'] = a_df['Zipcode'].fillna(a_df['Street-Number'].map(b_df.set_index('Street-Number')['Zipcode']))
print(a_df)
# Let's first load the sf_permits DF from scratch to overwrite any changes
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")
# Number and proportion of missing Zipcode values before...
# get the number of missing data points in the Zipcode column of the sf_permits data frame
sf_missing_zipcodes_count_before = sf_permits['Zipcode'].isnull().sum()
# total number of Zipcode rows
sf_total_zipcodes = len(sf_permits['Zipcode'])
# percent of missing zipcodes data in sf permits dataset
sf_missing_zipcodes_percent_before = (sf_missing_zipcodes_count_before / sf_total_zipcodes) * 100
# Add the new STREET column to the sf_permits data frame as explained above.
sf_permits['STREET'] = (sf_permits['Street Name'] + ' ' + sf_permits['Street Suffix']).str.upper().str.replace('AV', 'AVE')
oa_update_df = pd.DataFrame()
oa_update_df[['STREET', 'NUMBER', 'POSTCODE']] = oa_sf_df[['STREET', 'NUMBER', 'POSTCODE']]
sf_permits['STREET-NUMBER'] = sf_permits['STREET'] + '-' + sf_permits['Street Number'].astype(str) + sf_permits['Street Number Suffix'].str.upper().fillna('')
oa_update_df['STREET-NUMBER'] = oa_update_df['STREET'] + '-' + oa_update_df['NUMBER']
oa_update_df.drop_duplicates('STREET-NUMBER', inplace=True)
x = pd.Series([1,2,3], index=['one', 'two', 'three'])
y = pd.Series(['foo', 'bar', 'baz'], index=[1,2,3])
x.map(y)
x = pd.Series([0,2,3], index=['one', 'two', 'three'])
y = pd.Series(['foo', 'bar', 'baz'], index=[1,2,3])
x.map(y)
x = pd.Series([3,2,1], index=['one', 'two', 'three'])
y = pd.Series(['foo', 'bar', 'baz'], index=[1,2,3])
x.map(y)
sf_permits['Zipcode'] = sf_permits['Zipcode'].fillna(sf_permits['STREET-NUMBER'].map(oa_update_df.set_index('STREET-NUMBER')['POSTCODE']))
sf_permits[(sf_permits['Street Name'] == 'Washington') & (sf_permits['Street Number'] == 3191)]
# Number and proportion of missing Zipcode values after...
# get the number of missing data points in the Zipcode column of the sf_permits data frame
sf_missing_zipcodes_count_after = sf_permits['Zipcode'].isnull().sum()
# percent of missing zipcodes data in sf permits dataset
sf_missing_zipcodes_percent_after = (sf_missing_zipcodes_count_after / sf_total_zipcodes) * 100

print("BEFORE")
print(f"Zipcode rows: {len(sf_permits.Zipcode)}")
print(f"Missing zipcodes count: {sf_missing_zipcodes_count_before}")
print(f"Missing zipcodes %: {sf_missing_zipcodes_percent_before:.2f}%")
print("*" * 3)
print("AFTER")
print(f"Zipcode rows: {len(sf_permits.Zipcode)}")
print(f"Missing zipcodes count: {sf_missing_zipcodes_count_after}")
print(f"Missing zipcodes %: {sf_missing_zipcodes_percent_after:.2f}%")

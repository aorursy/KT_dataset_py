# load the libraries
import pandas as pd
import numpy as np

# read in all our data
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 
# Let's look at 10 observations to see the missing values
sf_permits.sample(10)
missing_values = sf_permits.isnull()
missing_values_count = missing_values.sum()

#display first 10 columns to see missing counts for each
missing_values_count[0:10]
total_cells = np.product(sf_permits.shape)   ## Total number of cells
total_missing = missing_values_count.sum()   ## Total number of missing cells
missing_perc = round((total_missing)/(total_cells)*100,2)
missing_perc
# look at the # of missing points in the first ten columns
missing_values_count[0:30]

# Remove rows and columns if it contains missing values using "sf_permits" data set
sf_permits.dropna()
columns_with_na_dropped = sf_permits.dropna(axis=1)
columns_with_na_dropped.head()
# See how much data we lost?
print("columns in original dataset: %d \n" % sf_permits.shape[1])
print("columns with NA's dropped: %d" % columns_with_na_dropped.shape[1])
# get a small subset of the NFL dataset
subset_sf_permits = sf_permits.loc[:, 'Permit Number' : 'Zipcode'].head()
subset_sf_permits.head()
# replace all NA's in the subset with 0
subset_sf_permits.fillna(0)
# replace all NA's with the value that comes directly after that empty cell in the same columns and replace remaining empty cells with 0
subset_sf_permits.fillna(method='bfill', axis=0).fillna("0")
sf_permits.dtypes.sample(10)

#from sklearn.preprocessing import Imputer
#sf_permits_imputer = Imputer()
#sf_permits_with_imputed_values = sf_permits_imputer.fit_transform(sf_permits)
encoded_sf_permits = pd.get_dummies(sf_permits)

features.iloc[:,10:].head(5)
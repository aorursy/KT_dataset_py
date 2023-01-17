# import numpy and pandas

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# read in the San Francisco building permits data
sfPermits = pd.read_csv("../input/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0)
sfPermits.sample(5)
# Calculate total number of cells in dataframe
totalCells = np.product(sfPermits.shape)

# Count number of missing values per column
missingCount = sfPermits.isnull().sum()

# Calculate total number of missing values
totalMissing = missingCount.sum()

# Calculate percentage of missing values
print("The SF Permits dataset contains", round(((totalMissing/totalCells) * 100), 2), "%", "missing values.")

missingCount[['Street Number Suffix', 'Zipcode']]

print("Percent missing data in Street Number Suffix column =", (round(((missingCount['Street Number Suffix'] / sfPermits.shape[0]) * 100), 2)))
print("Percent missing data in Zipcode column =", (round(((missingCount['Zipcode'] / sfPermits.shape[0]) * 100), 2)))
sfPermits.dropna()
sfPermitsCleanCols = sfPermits.dropna(axis=1)
sfPermitsCleanCols.head()
print("Columns in original dataset: %d \n" % sfPermits.shape[1])
print("Columns with na's dropped: %d" % sfPermitsCleanCols.shape[1])
imputeSfPermits = sfPermits.fillna(method='ffill', axis=0).fillna("0")

imputeSfPermits.head()
import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read the data

stores = pd.read_csv('/kaggle/input/sample-data/stores.csv')
stores.head(2)
stores.shape
# Add new columns

stores['NetProfit'] = stores.TotalSales - stores.OperatingCost
stores.head(2)
stores.shape
# '.assign' can be used to create new columns

stores = stores.assign(NetProfit1 = stores.TotalSales - stores.OperatingCost, NetProfit2 = stores.TotalSales - (stores.OperatingCost + stores.AcqCostPercust))
stores = stores.assign(np1 = stores.TotalSales - stores.OperatingCost, np2 = stores.TotalSales - stores.OperatingCost)
stores.head(2)
# Delete columns

# to drop the column, "Inplace = True" needs to be used

stores.drop(columns = 'np2', axis = 1, inplace = True)
stores.drop(columns = 'np1', axis = 1, inplace = True)
# Columsn 'np1' and 'np2' were dropped from original data set.

stores.head(1)
# Re-arrange the columns

stores[['TotalSales','StoreType', 'StoreName', 'Location']].head(3)
# Subsetting the columns ([], loc, iloc)

stores['StoreCode'].head(2)
# syntax for using loc and iloc (rows , columns) = (:,:)

# rows and columns given in range

stores.iloc[:,0:3].head(3)
stores.loc[:,['StoreCode', 'StoreName', 'StoreType']].head(3)
# Rename the columns 

# (here also we need to use "Inplace = True" in order to change the column name in original datset)

stores.rename

stores.rename(columns = {'TotalSales': 'Sales'}).head(2)
stores.head(1)
# Read the data

stores = pd.read_csv('/kaggle/input/sample-data/stores.csv')
stores.head(2)
# Filter

# Get records from stores where Location is Delhi

stores[stores.Location == 'Delhi']
# Sort

stores.sort_values('TotalSales', ascending = True)

# to sort values in descending, simple use 'ascending = False'
# Similarly we can sort based on more than one column

# stores.sort_values(['Location', 'TotalSales'], ascending = [True, False])
# Removal of duplicates

# Read the data

Score = pd.read_csv('/kaggle/input/sample-data1/Score.csv')
Score
# Duplicate values

Score[Score.duplicated()]
# pleas try below

# score.loc[score.duplicated(),:] 

# score.loc[-score.Student.duplicated(),:] 
# Data Imputation

# Read the data

stores = pd.read_csv('/kaggle/input/sample-data/stores.csv')
stores.shape[0]
# percentage of missing values from stores dataset

1 - stores.count()/stores.shape[0]
# Column 'AcqCostPercust' has missing values

data = stores.AcqCostPercust

data
# fill the missing values

# if null values are more, we can drop the column 'AcqCostPercust' using below

a = stores.dropna(axis = 1)
# we can fill the missing values using below

data.fillna(data.mean())
stores['New'] = stores.TotalSales.clip(lower = 100, upper = 400)
# Values are restricted to the range (100,400)

stores[['TotalSales', 'New']]
# Binning example

stores['Bins'] = pd.cut(stores.TotalSales, 10)
stores[['Bins', 'TotalSales'] ]
pd.cut(stores.TotalSales, range(50, 1000, 50))
import numpy as np
stores["Region"] = np.where((stores.Location == "Delhi"), "North",

            np.where((stores.Location == "Chennai"), "South",

                     np.where((stores.Location == "Kolkata"), "East",

                              np.where((stores.Location == "Mumbai"), "West", ""))))
stores[['Region', 'Location']]
# Groupby

stores.groupby('Location').TotalSales.agg(['sum', 'mean']).reset_index()
# /kaggle/input/sample-data-2/Transaction_Summary.csv

# /kaggle/input/sample-data-2/Demographic_Data.csv
# Merge

a = pd.read_csv('/kaggle/input/sample-data-2/Transaction_Summary.csv')

b = pd.read_csv('/kaggle/input/sample-data-2/Demographic_Data.csv')
a
b
# Inner join

pd.merge(left = a, right= b, right_on = 'CustName', left_on='CustomerName', how = 'inner')

# Similarly other joins can be performed
# Append

# Read the data

Q3 = pd.read_csv('/kaggle/input/sample-data-3/POS_Q3.csv')

Q1 = pd.read_csv('/kaggle/input/sample-data-3/POS_Q1.csv')

Q4 = pd.read_csv('/kaggle/input/sample-data-3/POS_Q4.csv')

Q2 = pd.read_csv('/kaggle/input/sample-data-3/POS_Q2.csv')
Q1.append([Q2, Q3, Q4], ignore_index = True)
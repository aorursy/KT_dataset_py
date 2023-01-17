# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

transactions = pd.read_csv('../input/transactions.csv')


#Summary of transaction data set
transactions.info()
#Numbers of Columns
transactions.shape[1]

# Numbers of Records
transactions.shape[0]
# Get the row names
transactions.index.values
#Get the column names
transactions.columns.values
#view top 10 records
transactions.head(10)
#Change the name of Column "Quantity" to "Quant"
transactions.rename(columns={'Quantity' :'Quant'})
#Change the name of columns ProductID and UserID to PID and UID respectively 
transactions.rename(columns ={"ProductID":"PID","UserID": "UID"})
#Order the rows of transactions by TransactionId decending
# if ascending then ascending = True,
transactions.sort_values('TransactionID', ascending=False)
# Order the rows of transactions by Quantity ascending, TransactionDate descending

transactions.sort_values(['Quantity','TransactionDate'],ascending=[True,False])
# Set the column order of transactions as ProductID, Quantity, TransactionDate, TransactionID, UserID
transactions[['ProductID', 'Quantity', 'TransactionDate', 'TransactionID', 'UserID']]
# Make UserID the first column of transactions
transactions[pd.unique(['UserID'] + transactions.columns.values.tolist()).tolist()]
#Extracting arrays from a DataFrame
# Get the 2nd column
transactions[:2]
transactions.iloc[1]
# Get the ProductID array
transactions.ProductID.values
#Get the productId array using a variable 
col= "ProductID"
transactions[[col]].values[:,0]
#Row Subsetting
#Subset rows 1,3 and 6
#transactions.iloc[[1-1,3-1,6-1]]
transactions.iloc[[0,2,5]]
#subset rows excluding 1,3, and 6
transactions.drop([0,2,5],axis=0)
#Subset the fist three rows
transactions[:3]
transactions.head(3)
# Subset the last 2 rows
transactions.tail(2)
#subset rows excluding the last 2 rows
transactions.head(-2)
# Subset rows excluding the first 3 rows
transactions[3:]
transactions.tail(-3)
#Subset rows where Quantity > 1
transactions[(transactions.Quantity >1 )]
#Subset rows where UserID =2
transactions[transactions.UserID ==2]
# Subset rows where Quantity > 1 and UserID = 2
transactions[(transactions.Quantity >0) & (transactions.UserID == 2)]
# Subset rows where Quantity + UserID is > 3
transactions[(transactions.Quantity + transactions.UserID )> 3]
#Subset rows where an external array,foo, is True
foo = np.array([True,False,True,False,True,False,True,False,True,False])

transactions[foo]
# Subset rows where an external array, bar, is positive
bar = np.array([1, -3, 2, 2, 0, -4, -4, 0, 0, 2])
bar
transactions[bar > 0]
# Subset rows where foo is TRUE or bar is negative
transactions[foo | (bar <0)]
# Subset the rows where foo is not TRUE and bar is not negative

transactions[~foo | bar >= 0]
#Column excercises
#Subset by columns 1 and 3
transactions.iloc[:,[0,2]]
## Subset by columns TransactionID and TransactionDate
transactions[['TransactionID','TransactionDate']]
# Subset by columns TransactionID and TransactionDate wtih logical operator
transactions.loc[transactions.TransactionID >5,['TransactionID','TransactionDate']]
# Subset columns by a variable list of columm names
cols = ["TransactionID","UserID","Quantity"]
transactions[cols]
# Subset columns excluding a variable list of column names
cols = ["TransactionID", "UserID", "Quantity"]
transactions.drop(cols,axis =1)
#Inserting and updating values
# Convert the TransactionDate column to type Date
transactions['TransactionDate'] = pd.to_datetime(transactions.TransactionDate)
transactions['TransactionDate']
# Insert a new column, Foo = UserID + ProductID
transactions['Foo'] = transactions.UserID + transactions.ProductID
transactions['Foo']
#post your query and comment.
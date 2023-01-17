import numpy as np

import pandas as pd
transactions = pd.read_csv('../input/transactions.csv')


transactions.info()
# How many rows?

transactions.shape[0]
# How many columns?

transactions.shape[1]
# Get the range of rows

transactions.index
# Get the row names

transactions.index.values
transactions.columns.values   # OR  transactions.columns
transactions.count()
transactions.sum(skipna=True, numeric_only=True)
transactions['TransactionID'].mean()
transactions.mean()
transactions['TransactionID'].mean()
transactions.median()
transactions.mode()
# Change the name of columns ProductID and UserID to PID and UID respectively

transactions.rename(columns={'ProductID': 'PID', 'UserID': 'UID'})  # use argument inplace=TRUE to keep the changes
# Set the column order of transactions as ProductID, Quantity, TransactionDate, TransactionID, UserID

transactions[['ProductID', 'Quantity', 'TransactionDate', 'TransactionID', 'UserID']]
# Make UserID the first column of transactions

transactions[pd.unique(['UserID'] + transactions.columns.values.tolist()).tolist()]
# Get the 2nd column

transactions.iloc[:, 2]
# Get the ProductID array

transactions.ProductID.values
# Get the ProductID array using a variable

col = "ProductID"

transactions[[col]].values[:, 0]
# Subset rows 1, 3, and 6

transactions.iloc[[0,2,5]]
# Subset rows exlcuding 1, 3, and 6

transactions.drop([0,2,5], axis=0)
# Subset the first 3 rows

transactions[:3]

transactions.head(3)
# Subset rows excluding the first 3 rows

transactions[3:]

transactions.tail(-3)
# Subset the last 2 rows

transactions.tail(2)
# Subset rows excluding the last 2 rows

transactions.tail(-2)
# Subset rows where Quantity > 1

transactions[transactions.Quantity > 1]
# Subset rows where UserID = 2

transactions[transactions.UserID == 2]
# Subset rows where Quantity > 1 and UserID = 2

transactions[(transactions.Quantity > 1) & (transactions.UserID == 2)]

# Subset rows where Quantity + UserID is > 3

transactions[transactions.Quantity + transactions.UserID > 3]
# Subset rows where an external array, foo, is True

foo = np.array([True, False, True, False, True, False, True, False, True, False])

transactions[foo]
# Subset rows where an external array, bar, is positive

bar = np.array([1, -3, 2, 2, 0, -4, -4, 0, 0, 2])

transactions[bar > 0]
# Subset rows where foo is TRUE or bar is negative

transactions[foo | (bar < 0)]
# Subset the rows where foo is not TRUE and bar is not negative

transactions[~foo & (bar >= 0)]
# Subset by columns 1 and 3

transactions.iloc[:, [0, 2]]
# Subset by columns TransactionID and TransactionDate

transactions[['TransactionID', 'TransactionDate']]
# Subset rows where TransactionID > 5 and subset columns by TransactionID and TransactionDate

transactions.loc[transactions.TransactionID > 5, ['TransactionID', 'TransactionDate']]
# Subset columns by a variable list of columm names

cols = ["TransactionID", "UserID", "Quantity"]

transactions[cols]
# Subset columns excluding a variable list of column names

cols = ["TransactionID", "UserID", "Quantity"]

transactions.drop(cols, axis=1)
# Convert the TransactionDate column to type Date

transactions['TransactionDate'] = pd.to_datetime(transactions.TransactionDate)
# Insert a new column, Foo = UserID + ProductID

transactions['Foo'] = transactions.UserID + transactions.ProductID
# Subset rows where TransactionID is even and set Foo = NA

transactions.loc[transactions.TransactionID % 2 == 0, 'Foo'] = np.nan
# Add 100 to each TransactionID

transactions.TransactionID = transactions.TransactionID + 100

transactions.TransactionID = transactions.TransactionID - 100  # revert to original IDs
# Insert a column indicating each row number

transactions['RowIdx'] = np.arange(transactions.shape[0])
# Insert columns indicating the rank of each Quantity, minimum Quantity and maximum Quantity

transactions['QuantityRk'] = transactions.Quantity.rank(method='average')

transactions['QuantityMin'] = transactions.Quantity.min()

transactions['QuantityMax'] = transactions.Quantity.max()
# Remove column Foo

transactions.drop('Foo', axis=1, inplace=True)
# Remove multiple columns RowIdx, QuantityRk, and RowIdx

transactions.drop(['QuantityRk', 'QuantityMin', 'QuantityMax'], axis=1, inplace=True)
# Grouping the rows of a DataFrame



#--------------------------------------------------

# Group By + Aggregate



# Group the transations per user, measuring the number of transactions per user

transactions.groupby('UserID').apply(lambda x: pd.Series(dict(

    Transactions=x.shape[0]

))).reset_index()
# Group the transactions per user, measuring the transactions and average quantity per user

transactions.groupby('UserID').apply(lambda x: pd.Series(dict(

    Transactions=x.shape[0],

    QuantityAvg=x.Quantity.mean()

))).reset_index()
# Joining DataFrames

# Load datasets from CSV

users = pd.read_csv('../input/users.csv')

sessions = pd.read_csv('../input/sessions.csv')

products = pd.read_csv('../input/products.csv')

transactions = pd.read_csv('../input/transactions.csv')
# Convert date columns to Date type

users['Registered'] = pd.to_datetime(users.Registered)

users['Cancelled'] = pd.to_datetime(users.Cancelled)

users
transactions['TransactionDate'] = pd.to_datetime(transactions.TransactionDate)

transactions
# Basic Joins



# Join users to transactions, keeping all rows from transactions and only matching rows from users (left join)

transactions.merge(users, how='left', on='UserID')
# Which transactions have a UserID not in users? (anti join)

transactions[~transactions['UserID'].isin(users['UserID'])]
# Join users to transactions, keeping only rows from transactions and users that match via UserID (inner join)

transactions.merge(users, how='inner', on='UserID')
# Join users to transactions, displaying all matching rows AND all non-matching rows (full outer join)

transactions.merge(users, how='outer', on='UserID')
# Determine which sessions occured on the same day each user registered

pd.merge(left=users, right=sessions, how='inner', left_on=['UserID', 'Registered'], right_on=['UserID', 'SessionDate'])
# Build a dataset with every possible (UserID, ProductID) pair (cross join)

df1 = pd.DataFrame({'key': np.repeat(1, users.shape[0]), 'UserID': users.UserID})

df2 = pd.DataFrame({'key': np.repeat(1, products.shape[0]), 'ProductID': products.ProductID})

pd.merge(df1, df2,on='key')[['UserID', 'ProductID']]
# Determine how much quantity of each product was purchased by each user

df1 = pd.DataFrame({'key': np.repeat(1, users.shape[0]), 'UserID': users.UserID})

df2 = pd.DataFrame({'key': np.repeat(1, products.shape[0]), 'ProductID': products.ProductID})

user_products = pd.merge(df1, df2,on='key')[['UserID', 'ProductID']]

pd.merge(user_products, transactions, how='left', on=['UserID', 'ProductID']).groupby(['UserID', 'ProductID']).apply(lambda x: pd.Series(dict(

    Quantity=x.Quantity.sum()

))).reset_index().fillna(0)
# For each user, get each possible pair of pair transactions (TransactionID1, TransactionID2)

pd.merge(transactions, transactions, on='UserID')
# Join each user to his/her first occuring transaction in the transactions table

pd.merge(users, transactions.groupby('UserID').first().reset_index(), how='left', on='UserID')
# Reshaping a data.table

# Read datasets from CSV

users = pd.read_csv('../input/users.csv')

transactions = pd.read_csv('../input/transactions.csv')
# Convert date columns to Date type

users['Registered'] = pd.to_datetime(users.Registered)

users
users['Cancelled'] = pd.to_datetime(users.Cancelled)

users
transactions['TransactionDate'] = pd.to_datetime(transactions.TransactionDate)

transactions
# Add column TransactionWeekday as Categorical type with categories Sunday through Saturday

transactions['TransactionWeekday'] = pd.Categorical(transactions.TransactionDate.dt.weekday_name, categories=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

transactions
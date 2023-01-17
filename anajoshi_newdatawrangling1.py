# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

transactions = pd.read_csv("../input/newdatawranglingds/transactions.csv")

users = pd.read_csv("../input/newdatawranglingds/users.csv")

sessions = pd.read_csv("../input/newdatawranglingds/sessions.csv")

products = pd.read_csv("../input/newdatawranglingds/products.csv")



#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

# Q1 Full summary of data

transactions.info()
# Q2 How many rows?

transactions.shape[0]
# Q3 How many columns?

transactions.shape[1]
# Q4 Get the row names as an array

transactions.index.values
# Q5 Get the column names as anarray

transactions.columns.values
# Q6 Change the name of column "Quantity" to "Quant"

transactions.rename(columns={'Quantity': 'Quant'})  # use argument inplace=TRUE to keep the changes
# Q7 Change the name of columns ProductID and UserID to PID and UID respectively

transactions.rename(columns={'ProductID': 'PID', 'UserID': 'UID'})  # use argument inplace=TRUE to keep the changes
# Q8 Order the rows of transactions by TransactionID descending

transactions.sort_values('TransactionID', ascending=False)
# Q9 Order the rows of transactions by Quantity ascending, TransactionDate descending

transactions.sort_values(['Quantity', 'TransactionDate'], ascending=[True, False])
# Q10 Set the column order of transactions as ProductID, Quantity, TransactionDate, TransactionID, UserID

transactions[['ProductID', 'Quantity', 'TransactionDate', 'TransactionID', 'UserID']]
# Q11 Make UserID the first column of transactions

transactions[pd.unique(['UserID'] + transactions.columns.values.tolist()).tolist()]
# Q12 Get the 2nd column

transactions.Quantity
# Q13 Get the ProductID array

transactions.ProductID.values
col = "ProductID"

transactions[[col]].values[:, 0]
# Q15 Subset rows 1, 3, and 6

transactions.iloc[[0,2,5]]
# Q16 Subset rows exlcuding 1, 3, and 6

transactions.drop([0,2,5], axis=0)
# Q17 Subset the first 3 rows

transactions[:3]

transactions.head(3)
# Q18 Subset rows excluding the first 3 rows

transactions[3:]

transactions.tail(-3)
# Q19 Subset the last 2 rows

transactions.tail(2)
# Q20 Subset rows excluding the last 2 rows

transactions.tail(-2)
# Q21 Subset rows where Quantity > 1

transactions[transactions.Quantity > 1]
# Q22 Subset rows where UserID = 2

transactions[transactions.UserID == 2]
# Q23 Subset rows where Quantity > 1 and UserID = 2

transactions[(transactions.Quantity > 1) & (transactions.UserID == 2)]
# Q24 Subset rows where Quantity + UserID is > 3

transactions[transactions.Quantity + transactions.UserID > 3]
foo = np.array([True, False, True, False, True, False, True, False, True, False])

transactions[foo]
# Q26 Subset rows where an external array, bar, is positive

bar = np.array([1, -3, 2, 2, 0, -4, -4, 0, 0, 2])

transactions[bar > 0]
# Q27 Subset rows where foo is TRUE or bar is negative

transactions[foo | (bar < 0)]
# Q28 Subset the rows where foo is not TRUE and bar is not negative

transactions[~foo & (bar >= 0)]
# Q29 Subset by columns 1 and 3

transactions.iloc[:, [0, 2]]
# Q30 Subset by columns TransactionID and TransactionDate

transactions[['TransactionID', 'TransactionDate']]
# Q31 Subset rows where TransactionID > 5 and subset columns by TransactionID and TransactionDate

transactions.loc[transactions.TransactionID > 5, ['TransactionID', 'TransactionDate']]
# Q32 Subset columns by a variable list of columm names

cols = ["TransactionID", "UserID", "Quantity"]

transactions[cols]
# Q33 Subset columns excluding a variable list of column names

cols = ["TransactionID", "UserID", "Quantity"]

transactions.drop(cols, axis=1)
# Q34 Convert the TransactionDate column to type Date

transactions['TransactionDate'] = pd.to_datetime(transactions.TransactionDate)
# Q35 Insert a new column, Foo = UserID + ProductID

transactions['Foo'] = transactions.UserID + transactions.ProductID
# Q36 Subset rows where TransactionID is even and set Foo = NA

transactions.loc[transactions.TransactionID % 2 == 0, 'Foo'] = np.nan
# Q37.   Add 100 to each TransactionID

transactions.TransactionID = transactions.TransactionID + 100

transactions.TransactionID = transactions.TransactionID - 100  # revert to original IDs
# Q38 Insert a column indicating each row number

transactions['RowIdx'] = np.arange(transactions.shape[0])
# Q39 Insert columns indicating the rank of each Quantity, minimum Quantity and maximum Quantity

transactions['QuantityRk'] = transactions.Quantity.rank(method='average')

transactions['QuantityMin'] = transactions.Quantity.min()

transactions['QuantityMax'] = transactions.Quantity.max()
# Q40 Remove column Foo

transactions.drop('Foo', axis=1, inplace=True)
# Q41 Remove multiple columns RowIdx, QuantityRk, and RowIdx

transactions.drop(['QuantityRk', 'QuantityMin', 'QuantityMax'], axis=1, inplace=True)
# Q42 Group the transations per user, measuring the number of transactions per user



transactions.groupby('UserID').apply(lambda x: pd.Series(dict(

                                                             Transactions=x.shape[0]

                                                              ))).reset_index()
transactions.groupby('UserID').apply(lambda x: pd.Series(dict(

                                                             Transactions=x.shape[0],

                                                             QuantityAvg=x.Quantity.mean()

                                                              ))).reset_index()
# Q44 Convert date columns to Date type

users['Registered'] = pd.to_datetime(users.Registered)

users['Cancelled'] = pd.to_datetime(users.Cancelled)

transactions['TransactionDate'] = pd.to_datetime(transactions.TransactionDate)
# Q45 Join users to transactions, keeping all rows from transactions and only matching rows from users (left join)

transactions.merge(users, how='left', on='UserID')
# Q46 Which transactions have a UserID not in users? (anti join)

transactions[~transactions['UserID'].isin(users['UserID'])]
# Q47 Join users to transactions, keeping only rows from transactions and users that match via UserID (inner join)

transactions.merge(users, how='inner', on='UserID')
# Q48 Join users to transactions, displaying all matching rows AND all non-matching rows (full outer join)

transactions.merge(users, how='outer', on='UserID')
# Q50 Build a dataset with every possible (UserID, ProductID) pair (cross join)



df1 = pd.DataFrame({'key': np.repeat(1, users.shape[0]), 'UserID': users.UserID})

df2 = pd.DataFrame({'key': np.repeat(1, products.shape[0]), 'ProductID': products.ProductID})

pd.merge(df1, df2,on='key')[['UserID', 'ProductID']]
# Q51 Determine how much quantity of each product was purchased by each user



df1 = pd.DataFrame({'key': np.repeat(1, users.shape[0]), 'UserID': users.UserID})

df2 = pd.DataFrame({'key': np.repeat(1, products.shape[0]), 'ProductID': products.ProductID})
user_products = pd.merge(df1, df2,on='key')[['UserID', 'ProductID']]
pd.merge(user_products, transactions, how='left', on=['UserID', 'ProductID']).groupby(['UserID', 'ProductID']).apply(lambda x: pd.Series(dict(Quantity=x.Quantity.sum()))).reset_index().fillna(0)
# Q52 For each user, get each possible pair of pair transactions (TransactionID1, TransactionID2)

pd.merge(transactions, transactions, on='UserID')
# Q53 Join each user to his/her first occuring transaction in the transactions table

pd.merge(users, transactions.groupby('UserID').first().reset_index(), how='left', on='UserID')
# Q56 Add column TransactionWeekday as Categorical type with categories Sunday through Saturday



transactions['TransactionWeekday'] = pd.Categorical(transactions.TransactionDate.dt.weekday_name, categories=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

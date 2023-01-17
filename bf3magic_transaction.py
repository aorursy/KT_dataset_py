# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATASET_PATH = "../input/onlinetransaction/OnlineTransaction.xlsx"
df = pd.read_excel(DATASET_PATH)
df.to_pickle("original_data")
print(df.shape)

df.head(3)
print("Number of Transactions: " + str(df.TransactionNO.nunique()))

print("Number of Customers: " + str(df.CustomerID.nunique()))

print("Number of Items: " + str(df.ItemID.nunique()))

print("Number of Description: " + str(df.Description.nunique()))
df.isnull().sum(axis=0)
data = df.dropna()

data.shape
print("Number of Transactions: " + str(data.TransactionNO.nunique()))

print("Number of Customers: " + str(data.CustomerID.nunique()))

print("Number of Items: " + str(data.ItemID.nunique()))

print("Number of Description: " + str(data.Description.nunique()))
data[data["Quantity"] < 0]
data = data[data["Quantity"] > 0]
data["ItemTotal"] = data["Quantity"] * data["UnitPrice"]
print(data.shape)

data.head(5)
data.to_pickle("NaN_Neg_Dropped")
data['TransactionTotal'] = data["ItemTotal"]
data.head(10)
TransactionTotals = data.groupby(data.TransactionNO).agg({'TransactionTotal': 'sum'})
print(TransactionTotals.shape)

TransactionTotals.head(3)
data = data.drop(columns = ["TransactionTotal"])

data_with_tTotal = data.merge(TransactionTotals, left_on='TransactionNO', right_on='TransactionNO')
data_with_tTotal.head(10)
data_with_tTotal["NumTransactions"] = data_with_tTotal.groupby('CustomerID')['TransactionNO'].transform('nunique')
data_with_tTotal.head(10)
df_avg_transaction_spend = data_with_tTotal[["TransactionNO", "CustomerID", "TransactionTotal", "NumTransactions"]].copy()

df_avg_transaction_spend.head(5)
df_avg_transaction_spend = df_avg_transaction_spend.drop_duplicates()

df_avg_transaction_spend.head(5)
df_customer_spend = df_avg_transaction_spend.groupby("CustomerID", as_index=False).agg({"TransactionTotal" : "mean"})
df_customer_spend[df_customer_spend["CustomerID"]==13047]
df_avg_transaction_spend[df_avg_transaction_spend["CustomerID"]==13047]
df1_customer_data = df_customer_spend.copy().rename(columns={"TransactionTotal": "AvgTransactionVolumn"})

print(df1_customer_data.shape)

df1_customer_data.head(10)
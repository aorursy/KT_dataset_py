import numpy as np 
import pandas as pd 
import seaborn as sns
pd.set_option('display.max_columns', None);
pd.set_option('display.max_rows', None);
pd.set_option('display.float_format', lambda x:'%.0f' % x)
import matplotlib.pyplot as plt
df = pd.read_csv('../input/online-retail-ii-uci/online_retail_II.csv')
df.head()
df['Description'].nunique()
df['Description'].value_counts().head()
df.groupby('Description').agg({'Quantity':'sum'}).head()
df.groupby('Description').agg({'Quantity':'sum'}).sort_values('Quantity', ascending = False).head()
# TOTAL INVOICE AMOUNT
df['Invoice'].nunique()
df['Total_Price'] = df['Quantity']*df['Price']
df.head(2)
df['InvoiceDate'].max()

df.groupby('Invoice').agg({'TotalPrice':'sum'}).head()
df.sort_values('Price', ascending = False).head()
df['Country'].value_counts()
df.groupby('Country').agg({'TotalPrice':'sum'}).head()
df.groupby('Country').agg({'TotalPrice':'sum'}).sort_values('TotalPrice', ascending = False).head()
df.isnull().sum()
df.dropna(inplace = True)
df.isnull().sum()
df.shape
df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T
for feature in ['Quantity','Price','TotalPrice']:
    Q1 = df[feature].quantile(0.01)
    Q3 = df[feature].quantile(0.99)
    IQR = Q3-Q1 
    upper = Q3 + 1.5*IQR
    lower = Q1 - 1.5*IQR
    
    if df[(df[feature] > upper) | (df[feature] < lower)].any(axis = None):
        print(feature, 'yes')
        print(df[(df[feature] > upper) | (df[feature] < lower)].shape[0])
    else:
        print(feature, 'No')
df.head()
df.info()
df['InvoiceDate'].min()
df['InvoiceDate'].max()
import datetime as dt
Today_time= dt.datetime(2011,12,9)
Today_time
df.info()
df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head()
df["Customer ID"] = df["Customer ID"].astype(int)
df["Customer ID"].head()
(Today_time-df.groupby("Customer ID").agg({"InvoiceDate":"max"})).head()
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv("../input/data.csv", encoding="ISO-8859-1")
df.head()
df.describe()
df.isnull().sum()
df_new = df.dropna(axis='rows', how='any')
df_new.describe()
df_new = df_new[(df_new['Quantity']>=0) & (df_new['UnitPrice']>0)]
df_new.describe()
df_new['amount_spent'] = df_new['Quantity'] * df_new['UnitPrice']
df_new.head()
df_new.isnull().sum()
df_new = df_new.groupby('Country', as_index=False).apply(lambda x: x.sort_values(by='amount_spent', ascending=False))
df_new.head()
df_new.dtypes
df_new['InvoiceDate'] = pd.to_datetime(df_new['InvoiceDate'])
df_new.head()
df_new['year'] = df_new['InvoiceDate'].dt.year

df_new['month'] = df_new['InvoiceDate'].dt.month

df_new['day'] = df_new['InvoiceDate'].dt.day
df_new['hour'] = df_new['InvoiceDate'].dt.hour

df_new['minute'] = df_new['InvoiceDate'].dt.minute

df_new['second'] = df_new['InvoiceDate'].dt.second
df_new.head()
plt.figure(figsize=(11,7))

ax = df_new.groupby(['year', 'month'])['Quantity'].sum().plot.bar()

for p in ax.patches: 

    ax.annotate(np.round(p.get_height(),decimals=2), 

                (p.get_x()+p.get_width()/2, p.get_height()), 

                ha='center', va='center', xytext=(0, 10), 

                textcoords='offset points')
plt.figure(figsize=(10,10))

df_new.groupby(['Country'])['Quantity'].sum().sort_values().plot.barh(title="Country v/s Quantity Purchased")
plt.figure(figsize=(10,10))

df_new.groupby(['Country'])['amount_spent'].sum().sort_values().plot.barh(title="Country v/s Amount Spent")
plt.figure(figsize=(10,10))

df_new[(df_new['Country']!='United Kingdom')].groupby(['Country'])['Quantity'].sum().sort_values().plot.barh(title="Country v/s Quantity Purchased w/o UK")
plt.figure(figsize=(10,10))

df_new[(df_new['Country']!='United Kingdom')].groupby(['Country'])['amount_spent'].sum().sort_values().plot.barh(title="Country v/s Amount Spent w/o UK")
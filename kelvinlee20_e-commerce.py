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
import pandas as pd

df = pd.read_csv('/kaggle/input/ecommerce-data/data.csv', encoding = "ISO-8859-1")
df
df.describe()
df.info()
df.isnull().sum()
df[df['CustomerID'].isnull()]
import matplotlib.pyplot as plt

no_customer_id = len(df[df['CustomerID'].isnull()].index)
have_customer_id = len(df[~df['CustomerID'].isnull()].index)

plt.bar(['No Customer ID','Have Customer ID'],[no_customer_id, have_customer_id])
def addAnnotateNumber(ax, fontsize=12, isFloat=False):
    for child in ax.patches:
        if child.get_height() > 0:
            if isFloat:
                ax.annotate(round(child.get_height(),3), 
                        (child.get_x()+child.get_width()/2, child.get_height()), 
                        fontsize=fontsize,ha='center')
            else:
                ax.annotate(int(child.get_height()), 
                        (child.get_x()+child.get_width()/2, child.get_height()), 
                        fontsize=fontsize,ha='center')
        else:
            ax.annotate(0, 
                        (child.get_x()+child.get_width()/2, 0), 
                        fontsize=fontsize,ha='center')
s = df.Country.value_counts()
plt.figure(figsize=(18,5))
ax = s[:19].plot.bar()
ax.set_xticklabels(labels=s.index[:19], rotation=45, ha='right')
addAnnotateNumber(ax)
plt.figure(figsize=(18,5))
ax = s[19:].plot.bar()
ax.set_xticklabels(labels=s.index[19:], rotation=45, ha='right')
addAnnotateNumber(ax)
df.InvoiceDate = pd.to_datetime(df.InvoiceDate)
df
df.InvoiceDate.hist(bins=30)
df.index = df.InvoiceDate
s = df.resample('M')['InvoiceNo'].nunique()
s
s.index = s.index.strftime('%Y-%m-%d')
plt.figure(figsize=(18,5))
ax = s.plot.bar()
ax.set_xticklabels(labels=s.index, rotation=45, ha='right')
addAnnotateNumber(ax)
df['Price'] = df['Quantity']*df['UnitPrice']
df
df1 = df.groupby('InvoiceNo').aggregate({'Price':'sum','InvoiceDate':'first'})
df1
df1 = df1.reset_index()
df1['isCancel'] = df1['InvoiceNo'].str.contains('C')
df1
len(df1[df1['isCancel']])
len(df1[~df1['isCancel']])
labels = 'Cancel Order', 'Order',
sizes = [3836, 22064]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.boxplot(df1[~df1.isCancel]['Price'])
ax2.boxplot(df1[~df1.isCancel]['Price'], showfliers=False)
ax1.set_ylabel("Price")
ax1.set_xlabel("Order")
ax2.set_ylabel("Price")
ax2.set_xlabel("Order")
plt.show()
df1[(~df1.isCancel)&(df1.Price<0)]
df1['InvoiceNo'].str.slice(stop=1).value_counts()
df1[df1['InvoiceNo'].str.slice(stop=1)=='A']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.boxplot(df1[df1.isCancel]['Price'])
ax2.boxplot(df1[df1.isCancel]['Price'], showfliers=False)
ax1.set_ylabel("Price")
ax1.set_xlabel("Cancel Order")
ax2.set_ylabel("Price")
ax2.set_xlabel("Cancel Order")
plt.show()
df2 = df1[df1['InvoiceNo'].str.slice(stop=1)=='5']
df2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
df2[df2['Price']<1000]['Price'].hist(bins=100, ax=ax1)
df2[df2['Price']<500]['Price'].hist(bins=100, ax=ax2)
ax1.set_xlabel("Price")
ax2.set_xlabel("Price")
df2[df2['Price']==0]
df3 = df2[df2['Price']!=0]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
df3[df3['Price']<5000]['Price'].hist(bins=100, ax=ax1)
df3[df3['Price']<1000]['Price'].hist(bins=100, ax=ax2)
df3[df3['Price']<500]['Price'].hist(bins=100, ax=ax3)
ax1.set_xlabel("Price")
ax2.set_xlabel("Price")
ax3.set_xlabel("Price")
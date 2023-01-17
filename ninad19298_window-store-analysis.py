import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
raw_data=pd.read_csv('/kaggle/input/windows-store/msft.csv')
raw_data.head(10)
raw_data.tail(10)
raw_data.describe(include='all')
raw_data.isnull().sum()
data=raw_data.dropna()
data.describe(include='all')
data['Rating'].value_counts()
ae=sns.countplot(x='Rating',data=data)
data['Category'].value_counts()
plt.figure(figsize=(35,5))

sns.countplot(x='Category',data=data)
data['Price'].value_counts()
data['Price'].describe()
### Eliminating signs

data['Price']=data['Price'].str.replace(',', '')

data['Price']=data['Price'].str.replace('₹', '')
data['Price'].tail()
### amaking Free=0

def con(x):

    if x=='Free':

        return 0

    else:

        return x
data['Price']=data['Price'].apply(lambda x:con(x))
data['Price']=pd.to_numeric(data['Price'],errors='coerce')
data['Price']=data['Price'].astype(int)
data['Price'].tail()
r = [-2,0, 100, 200, 500, 1000,2000,5000,10000]

g = ['free','0-100','100-200','200-500','500-1000','1000-2000','2000-5000','>5000']

data['price_band'] = pd.cut(data['Price'], bins=r, labels=g)
data['price_band'].value_counts()
price_non_Zero=data[data['Price']>0]

price_non_Zero.head()
plt.figure(figsize=(14,5))

sns.distplot(price_non_Zero['Price'],kde=False)
data.groupby('Rating')['No of people Rated'].describe()
plt.figure(figsize=(10,5))

sns.boxplot(x='Rating',y='No of people Rated',data=data)

data['Category'].unique()
plt.figure(figsize=(30,5))

sns.barplot(x='Category',y='Rating',data=data)

plt.ylabel("Average rating")
plt.figure(figsize=(34,5))

sns.violinplot(y='Rating',x='Category',data=data)

plt.ylabel("Average Rating")
plt.figure(figsize=(30,5))

sns.barplot(x='price_band',y='Rating',data=data)

plt.ylabel("Average Rating")
plt.figure(figsize=(30,5))

sns.violinplot(x='price_band',y='Rating',data=data)
plt.figure(figsize=(40,5))

sns.barplot(x='Category',y='No of people Rated',data=data)

plt.ylabel("Average No of people rated")
plt.figure(figsize=(40,5))

sns.violinplot(x='Category',y='No of people Rated',data=data)
plt.figure(figsize=(30,5))

sns.stripplot(x='Category',y='Price',data=data,jitter=True)
plt.figure(figsize=(20,5))

sns.countplot(x=data['Price']==0,hue=data['Category'])
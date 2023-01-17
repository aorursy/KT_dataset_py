# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/windows-store/msft.csv')
data.head()
data.describe()
data.isnull().sum()
sns.heatmap(data.isnull(),cmap='Blues')
data.dropna(inplace=True)
data['Price']=data['Price'].str.replace(',', '')

data['Price']=data['Price'].str.replace('₹', '')
def con(x):

    if x=='Free':

        return 0

    else:

        return x
data['Price']=data['Price'].apply(lambda x:con(x))
data['Price']=pd.to_numeric(data['Price'],errors='coerce')
data
data['Rating'].value_counts()
plt.figure(figsize=(14,5))

sns.countplot(x=data['Rating'],palette="spring")
data['Price'].value_counts()
data['Price'].plot()
non_zero_price=data[data['Price']>0]

non_zero_price.head()
plt.figure(figsize=(14,5))

sns.distplot(non_zero_price['Price'],kde=False)
data['Category'].value_counts()
plt.figure(figsize=(12,7))

sns.countplot(y=data['Category'],palette="Blues")
data.groupby('Rating')['No of people Rated'].describe()
plt.figure(figsize=(10,5))

sns.boxplot(x='Rating',y='No of people Rated',data=data)
plt.figure(figsize=(24,5))

sns.violinplot(y='Rating',x='Category',data=data)
plt.figure(figsize=(10,5))

sns.stripplot(x='Rating',y='Price',data=data,jitter=True)
plt.figure(figsize=(24,5))

sns.violinplot(x='Category',y='No of people Rated',data=data)
non_zero_price=data[data['Price']>0]
plt.figure(figsize=(10,5))

sns.stripplot(y='Category',x='Price',data=non_zero_price,jitter=True)
zero_price=data[data['Price']==0]
plt.figure(figsize=(10,5))

sns.countplot(x='Price',data=zero_price,hue='Category')

plt.legend(bbox_to_anchor=(1.05,1), loc=2 , borderaxespad=0. )
sns.heatmap(data.corr(),cmap='Blues')
sns.pairplot(data)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Data = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

Data.head()
Data.describe()
Data.dtypes
Data.columns
Data.tail()
Data.count()
Data.isnull().sum()
Data.isna().sum()
Data.shape
Data['shop_id'].mean()
Data['item_id'].mean()
Data['shop_id'].median()
Data['item_id'].median()
sns.lmplot(x='shop_id',y='item_id',data=Data,legend=True,palette='red')
sns.distplot(Data['shop_id'])
sns.distplot(Data['shop_id'],kde=False)
sns.distplot(Data['shop_id'],bins=3)
sns.distplot(Data['shop_id'],bins=2)
sns.distplot(Data['shop_id'],bins=5)
sns.distplot(Data['shop_id'],bins=10)
sns.distplot(Data['item_id'])
sns.distplot(Data['item_id'],kde=False)
sns.distplot(Data['item_id'],bins=5)
sns.distplot(Data['item_id'],bins=3)
sns.distplot(Data['item_id'],bins=10)
sns.countplot(x='shop_id',data=Data)
sns.countplot(x='item_id',data=Data)
pd.crosstab(index=Data['shop_id'],columns=Data['item_id'],dropna=True)
pd.crosstab(index=Data['item_id'],columns=Data['shop_id'],dropna=True)
sns.boxplot(x=Data['shop_id'])
sns.boxplot(y=Data['shop_id'])
sns.boxplot(x=Data['item_id'])
sns.boxplot(y=Data['item_id'])
sns.boxplot(x=Data['shop_id'],y=Data['item_id'])
plt.scatter(Data['shop_id'],Data['item_id'],c='green')

plt.title('Scatter plot of shop_id vs item_id')

plt.xlabel('shop_id')

plt.ylabel('item_id')



plt.hist(Data['shop_id'])
plt.hist(Data['item_id'])
plt.hist(Data['shop_id'],color='red',edgecolor='red',bins = 200)
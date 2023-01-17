# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

ecom_data = pd.read_csv('../input/ecommerce-data/data.csv', encoding='latin1')


ecom_data.head()

ecom_data.describe()
ecom_data.info()
ecom_data['InvoiceNo'].value_counts()
purchase_size = ecom_data['InvoiceNo'].value_counts()

purchase_size.describe()
sns.distplot(purchase_size.sample(frac=0.25))
sns.boxplot(purchase_size.sample(frac=0.25))
purchase_size.quantile(0.5)
ecom_data['StockCode'].value_counts()
ecom_data['StockCode'].value_counts().describe()
sns.distplot(ecom_data['StockCode'].value_counts())
sns.boxplot(ecom_data['StockCode'].value_counts())
sns.distplot(ecom_data['Quantity'])
sns.boxplot(ecom_data['Quantity'])
ecom_data[ecom_data['Quantity'] < -1000]
print('First instance')

print(ecom_data[(ecom_data['StockCode'] == '84347') & (ecom_data['CustomerID'] == 15838)])

print('Second Instance')

print(ecom_data[(ecom_data['StockCode'] == '23166') & (ecom_data['CustomerID'] == 12346)])

print('Third Instance')

print(ecom_data[(ecom_data['StockCode'] == '47566B') & (ecom_data['CustomerID'] == 15749)])

print(ecom_data[(ecom_data['StockCode'] == '85123A') & (ecom_data['CustomerID'] == 15749)])

print('Fourth Instance')

print(ecom_data[(ecom_data['StockCode'] == '22920') & (ecom_data['CustomerID'] == 16938)])
ecom_data[ecom_data['StockCode'] == '23166']
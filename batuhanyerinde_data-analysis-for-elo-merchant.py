# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data1 = pd.read_csv('../input/merchants.csv')

data2 = pd.read_csv('../input/new_merchant_transactions.csv')

data3 = pd.read_csv('../input/test.csv')

data4 = pd.read_csv('../input/train.csv')

data5 = pd.read_csv('../input/sample_submission.csv')

data1.info()
data2.info()
# correlation map

f,ax = plt.subplots(figsize = (18,18))

sns.heatmap(data1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,center = 0,vmax = 1,vmin = -0.2)

plt.show()
# correlation map

f,ax = plt.subplots(figsize = (18,18))

sns.heatmap(data2.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,center = 0,vmax = 1,vmin = -0.2)

plt.show()
data1.head()
data1.tail()
data2.head()
data2.tail()
data1.plot(kind='scatter', x='avg_sales_lag3', y='avg_sales_lag6',alpha = 0.9,color = 'green',figsize=(16,16))

plt.show()
data1.plot(kind='scatter', x='avg_sales_lag3', y='avg_sales_lag12',alpha = 0.9,color = 'red',figsize=(16,16))

plt.show()
data1.plot(kind='scatter', x='avg_sales_lag6', y='avg_sales_lag12',alpha = 0.9,color = 'blue',figsize=(16,16))

plt.show()
data2.columns
data2.plot(kind='scatter', x='purchase_amount', y='installments',alpha = 0.9,color = 'orange',figsize=(16,16))

plt.show()
#data2.plot(kind='scatter', x='card_id', y='city_id',alpha = 0.9,color = 'orange',figsize=(16,16))

#plt.show()
data1.info()
print(data1['merchant_id'].value_counts(dropna =False))
print(data1['category_4'].value_counts(dropna =False))
data2.info()
print(data2['authorized_flag'].value_counts(dropna =False))
print(data2['city_id'].value_counts(dropna =False))
print(data2['card_id'].value_counts(dropna =False))
print(data2['category_3'].value_counts(dropna =False))
data1.describe()
data2.describe()
data2.boxplot(column='purchase_amount')
data_new1 = data1.head()

data_new1
melted1 = pd.melt(frame = data_new1,id_vars = 'merchant_id',value_vars = ['merchant_category_id','subsector_id'])

melted1
data_new2 = data2.head()

data_new2
melted2 = pd.melt(frame = data_new2,id_vars = 'card_id',value_vars = ['purchase_amount','installments'])

melted2
# Reverse of melting (pivoting_data)

melted1.pivot(index = 'merchant_id',columns = 'variable',values = 'value')
# concetenating data 

conc_data = data1.head()

conc_data1 = data1.tail()

conc_data_row = pd.concat([conc_data,conc_data1],axis = 0,ignore_index = False)

conc_data_row
data_0 = data1['merchant_category_id'].head()

data_1 = data1['subsector_id'].head()

conc_data_col = pd.concat([data_0,data_1],axis = 1,ignore_index = False)

conc_data_col
data1['avg_sales_lag3'].fillna('empty',inplace = True)
data1.head()
data1['avg_sales_lag6'].fillna(0.0,inplace = True)
assert data1['avg_sales_lag6'].notnull().all()
data1['avg_sales_lag6'] = data1['avg_sales_lag6'].astype('int64')
data1.info()
data1.columns
data2.columns
data1['category_1'] = data1['category_1'].astype("category")
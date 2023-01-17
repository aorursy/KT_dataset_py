# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/merchants.csv')
print(data.shape)

print(data.dtypes)
data.head()
data['merchant_group_id'] = data['merchant_group_id'].astype(object)

data['merchant_category_id'] = data['merchant_category_id'].astype(object)

data['subsector_id'] = data['subsector_id'].astype(object)

data['city_id'] = data['city_id'].astype(object)

data['state_id'] = data['state_id'].astype(object)

data['category_2'] = data['category_2'].astype(object)
print(data.dtypes)
data.head()
print(data['numerical_1'].describe())
print(data['numerical_2'].describe())
f, ax = plt.subplots(figsize=(10, 8))

sns.distplot(data['numerical_1'])
f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(data.numerical_1)
f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(data.numerical_2)
fig, ax = plt.subplots(1, 5, figsize = (16, 6));

data['category_1'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='teal', title='Anonymized_Category_1')

data['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[1], color='gold', title='most_recent_sales_range')

data['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[2], color='black', title='most_recent_purchases_range')

data['category_4'].value_counts().sort_index().plot(kind='bar', ax=ax[3], color='blue', title='Anonymized_Category_4')

data['category_2'].value_counts().sort_index().plot(kind='bar', ax=ax[4], color='red', title='Anonymized_Category_2')
print(data.dtypes)
data.head()
#### 3 month, 6 month, 12 month Analysis

#1 Sales

print(data['avg_sales_lag3'].describe())

print(data['avg_sales_lag6'].describe())

print(data['avg_sales_lag12'].describe())

#avg_sales_lag3

#avg_sales_lag6

#avg_sales_lag12

data[data.avg_sales_lag3.isnull()==1].head(15)
fig, ax = plt.subplots(1, 3, figsize = (16, 6));

data['avg_sales_lag3'].value_counts().sort_index().plot(kind='hist', ax=ax[0], color='teal', title='3_Month');

data['avg_sales_lag6'].value_counts().sort_index().plot(kind='hist', ax=ax[1], color='brown', title='6_Month');

data['avg_sales_lag12'].value_counts().sort_index().plot(kind='hist', ax=ax[2], color='blue', title='12_Month');

print(data[data.avg_sales_lag3>10].shape[0])

print(data[data.avg_sales_lag6>10].shape[0])

print(data[data.avg_sales_lag12>10].shape[0])

# 최근

print((data[(data.avg_sales_lag3>10)&(data.most_recent_sales_range!="E")].shape)[0])

print((data[(data.avg_sales_lag6>10)&(data.most_recent_sales_range!="E")].shape)[0])

print((data[(data.avg_sales_lag12>10)&(data.most_recent_sales_range!="E")].shape)[0])

f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='most_recent_sales_range', y="avg_sales_lag3", data=data)

plt.xlabel("most_recent_sales_range")

plt.ylabel("avg_sales_lag3")

plt.show
f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='most_recent_sales_range', y="avg_sales_lag6", data=data)

plt.xlabel("most_recent_sales_range")

plt.ylabel("avg_sales_lag3")

plt.show
f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='most_recent_sales_range', y="avg_sales_lag12", data=data)

plt.xlabel("most_recent_sales_range")

plt.ylabel("avg_sales_lag3")

plt.show


#### 3 month, 6 month, 12 month Analysis

#2 Purchases

print(data['avg_purchases_lag3'].describe())

print(data['avg_purchases_lag6'].describe())

print(data['avg_purchases_lag12'].describe())

#avg_sales_lag3

#avg_sales_lag6

#avg_sales_lag12

data[data==np.inf].count()
data.loc[data['avg_purchases_lag3']==np.inf,].head(15)
#data2 : not infinite variables in avg_purchases_lag

data2 = data.loc[data["avg_purchases_lag3"]!=np.inf,]

print(data2['avg_purchases_lag3'].describe())

print(data2['avg_purchases_lag6'].describe())

print(data2['avg_purchases_lag12'].describe())        
fig, ax = plt.subplots(1, 3, figsize = (16, 6));

data2['avg_purchases_lag3'].value_counts().sort_index().plot(kind='hist', ax=ax[0], color='teal', title='3_Month');

data2['avg_purchases_lag6'].value_counts().sort_index().plot(kind='hist', ax=ax[1], color='brown', title='6_Month');

data2['avg_purchases_lag12'].value_counts().sort_index().plot(kind='hist', ax=ax[2], color='blue', title='12_Month');

print(data[data.avg_purchases_lag3>10].shape[0])

print(data[data.avg_purchases_lag6>10].shape[0])

print(data[data.avg_purchases_lag12>10].shape[0])

data3 = data2[data2.avg_purchases_lag12>10]

fig, ax = plt.subplots(1, 3, figsize = (16, 6));

data3['avg_purchases_lag3'].value_counts().sort_index().plot(kind='hist', ax=ax[0], color='teal', title='3_Month');

data3['avg_purchases_lag6'].value_counts().sort_index().plot(kind='hist', ax=ax[1], color='brown', title='6_Month');

data3['avg_purchases_lag12'].value_counts().sort_index().plot(kind='hist', ax=ax[2], color='blue', title='12_Month');



#### 3 month, 6 month, 12 month Analysis

#3 Purchases

print(data['active_months_lag3'].describe())

print(data['active_months_lag6'].describe())

print(data['active_months_lag12'].describe())

fig, ax = plt.subplots(1, 3, figsize = (16, 6));

data['active_months_lag3'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='teal', title='3_Month');

data['active_months_lag6'].value_counts().sort_index().plot(kind='bar', ax=ax[1], color='brown', title='6_Month');

data['active_months_lag12'].value_counts().sort_index().plot(kind='bar', ax=ax[2], color='blue', title='12_Month');

x1 = data.loc[data['active_months_lag3']!=3]

x2 = data.loc[data['active_months_lag6']!=6]

x3 = data.loc[data['active_months_lag12']!=12]



fig, ax = plt.subplots(1, 3, figsize = (16, 6));

x1['active_months_lag6'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='teal', title='3_Month');

x2['active_months_lag6'].value_counts().sort_index().plot(kind='bar', ax=ax[1], color='brown', title='6_Month');

x3['active_months_lag12'].value_counts().sort_index().plot(kind='bar', ax=ax[2], color='blue', title='12_Month');

data["consistency"] = "Y"

data.loc[data["active_months_lag12"]!=12,"consistency"]= "N"



data.head()
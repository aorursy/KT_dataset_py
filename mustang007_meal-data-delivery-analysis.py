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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/train.csv')
meal = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/meal_info.csv')
center = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/fulfilment_center_info.csv')
print(train.head())
print(meal.head())
print(center.head())
merge1 = pd.merge(train, center, on='center_id', how='inner')
merge1
data = pd.merge(merge1, meal, on='meal_id')
data.head()
data.isnull().sum()
week = data.groupby(['week'])[['num_orders']].sum()
week
plt.figure(figsize=(20,5))
sns.lineplot(x=week.index, y =week['num_orders'], color='cyan')
plt.title('Total order per week')
# plt.xticks(rotation='90')
plt.show()
center = data.groupby(['center_id'])[['num_orders']].sum()
center
plt.figure(figsize=(25,5))
sns.barplot(x=center.index, y=center['num_orders'],palette="deep")
plt.xticks(rotation='45')
plt.show()
meal = data.groupby(['meal_id'])[['num_orders']].sum()
meal
plt.figure(figsize=(25,5))
sns.barplot(x=meal.index, y=meal['num_orders'],palette="deep")
plt.xticks(rotation='45')
plt.title('Total number of order per meal_id')
plt.show()
meal_checkout_price = data.groupby(['meal_id'])[['checkout_price']].mean()
meal_checkout_price
plt.figure(figsize=(25,5))
sns.barplot(x=meal_checkout_price.index, y=meal_checkout_price['checkout_price'],palette="deep")
plt.xticks(rotation='45')
plt.title('Average checkout price per meal_id')
plt.show()
city = data.groupby(['city_code'])[['num_orders']].sum()
city
plt.figure(figsize=(25,5))
sns.barplot(x=city.index, y=city['num_orders'])
plt.xticks(rotation='45')
plt.title('Total number of ordres per city_code')
plt.show()
region = data.groupby(['region_code'])[['num_orders']].sum()
region
plt.figure(figsize=(15,5))
sns.barplot(x=region.index, y=region['num_orders'])
plt.title('Total number of ordres per region')
plt.show()
print('total unique item',data['center_type'].nunique())
data.groupby(['center_type'])[['center_type']].count()
# Total number of orders per week for 
center_type = ['TYPE_A','TYPE_B','TYPE_C']
for i in center_type:
    a = data[data['center_type']==i].groupby(['week'])[['num_orders']].sum()
    plt.figure(figsize=(20,5))
    sns.lineplot(x=a.index, y =a['num_orders'], color='cyan')
    plt.title('Total number of orders for {} per week'.format(i))
    # plt.xticks(rotation='90')
    plt.show()

a = data['category'].unique()
for i in a:
    a = data[data['category']==i].groupby(['week'])[['num_orders']].sum()
    plt.figure(figsize=(20,5))
    sns.lineplot(x=a.index, y =a['num_orders'], color='cyan')
    plt.title('Total number of orders for {} per week'.format(i))
    plt.show()
a = data['cuisine'].unique()
a = data.groupby(['category','cuisine'])['num_orders'].sum()

a  = a.unstack().fillna(0)
a.plot(kind='bar', figsize=(20,5))
a = data.groupby(['center_type','cuisine'])['num_orders'].sum()
a = a.unstack().fillna(0)
a.plot(kind='bar')
a = data['cuisine'].unique()
for i in a:
    a = data[data['cuisine']==i].groupby(['week'])[['num_orders']].sum()
    plt.figure(figsize=(20,5))
    sns.lineplot(x=a.index, y =a['num_orders'], color='cyan')
    plt.title('Total number of orders for {} per week'.format(i))
    plt.show()
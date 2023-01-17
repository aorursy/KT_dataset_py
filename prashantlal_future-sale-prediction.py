# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
items=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
sales_train=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
item_categories=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
test=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
shops=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
sample_submission=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items.head()
items.info()
items['item_id'].nunique()
items['item_category_id'].nunique()
sales_train.head()
sales_train.info()
sales_train['date']=pd.DatetimeIndex(sales_train['date'])
sales_train.info()
print(min(sales_train['date']))
print(max(sales_train['date']))
print(sales_train['shop_id'].nunique())
print(sales_train['item_id'].nunique())
sales_train.isnull().any()
item_categories.head()
item_categories['item_category_id'].nunique()
item_categories.isnull().any()
test.head()
test.isnull().any()
shops.head()
shops['shop_id'].nunique()
sample_submission.head()
df=pd.merge(sales_train, items, on='item_id', how='inner')
df.head()
def positive_check(df,col):
    if df[df[col]<0].shape[0]>0:
        print('Have negative values')
    else:
        print("Don't have negative values")
for var in ['item_id', 'shop_id', 'item_category_id']:
    print(var, positive_check(df, var))
plt.figure(figsize=(10,10))
sns.boxplot(df['item_price'])
plt.show
plt.hist(df[df['item_price']<30000]['item_price'], bins=30)
plt.hist(df[df['item_price']<10000]['item_price'], bins=30)
plt.hist(df[df['item_price']<6000]['item_price'], bins=30)
index=df[df['item_price']>40000].index
df.drop(index, axis=0, inplace=True)
plt.figure(figsize=(10,10))
sns.boxplot(df['item_price'])
plt.show
df.head()
pd.options.display.max_columns = None
pd.options.display.max_rows = None
df.groupby('item_category_id').agg(mean=('item_price','mean'), median=('item_price','median'),
                                   std=('item_price','std'), min=('item_price','min'),
                                   max=('item_price','max'), count=('item_price','count'))
df['item_price'].replace(-1,0,inplace=True)
table_1=df.groupby('item_category_id').agg(mean=('item_price','mean'), median=('item_price','median'),
                                   std=('item_price','std'), min=('item_price','min'),
                                   max=('item_price','max'), count=('item_price','count'))
table_1
df=pd.merge(df, table_1[['mean','median','std']], on='item_category_id', how='inner')
df.head()
plt.hist(df['item_cnt_day'], bins=30)
sns.boxplot(df['item_cnt_day'])
first_quantile=df['item_cnt_day'].quantile(0.25)
third_quantile=df['item_cnt_day'].quantile(0.75)
iqr=third_quantile-first_quantile
print(first_quantile,third_quantile,iqr)

df['item_cnt_day'].quantile(np.arange(0,1.01,0.01))
index=df[df['item_cnt_day']>5].index
len(index)
len(index)/df.shape[0]*100
df.drop(index, axis=0, inplace=True)
sns.boxplot(df['item_cnt_day'])
def correction(x):
    if x<0:
        return 0
    else:
        return x
df['item_cnt_day']=df['item_cnt_day'].apply(lambda x:correction(x))
sns.boxplot(df['item_cnt_day'])
df.head()
table_2=df.groupby('shop_id').agg(cat_count=('item_category_id', 'nunique'),
                         item_count=('item_id', 'count'))
table_2['item_per_cat']=round(table_2['item_count']/table_2['cat_count'])
table_2
table_3=df.groupby(['shop_id','date_block_num']).agg(cat_count=('item_category_id', 'nunique'),
                         item_count=('item_id', 'count'))
table_4=table_3.groupby('shop_id').agg(avg_cat_count=('cat_count','mean'),
                                    avg_item_count=('item_count','mean')).round()
table_4['avg_item_per_cat']=round(table_4['avg_item_count']/table_4['avg_cat_count'])
table_4
df.head()
df=pd.merge(df,table_4['avg_item_per_cat'], how='inner', on='shop_id')
df.head()
df.drop(['date','date_block_num','item_name','item_category_id','item_price'], axis=1, inplace=True)
df.head()
df.sort_values(by=['shop_id','item_id'], inplace=True)
df.head()
test.head(10)

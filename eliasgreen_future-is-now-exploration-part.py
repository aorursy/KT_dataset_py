import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

sample_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items.head(5)
items.info()
items.describe()
sns.set(rc={'figure.figsize':(11.7,12)})

ax = sns.countplot(y = 'item_category_id',

              data = items,

              order = items['item_category_id'].value_counts(ascending=True).index)
print('Count of categories with count of items > 1000 is', len(list(filter(lambda x: x > 1000, items['item_category_id'].value_counts(ascending=True)))))
print('Count of categories with count of items 100-1000 is', len(list(filter(lambda x: 100 <= x <= 1000, items['item_category_id'].value_counts(ascending=True)))))
print('Count of categories with count of items < 100 is', len(list(filter(lambda x: x < 100, items['item_category_id'].value_counts(ascending=True)))))
item_categories.head(5)
item_categories.info()
item_categories.describe()
shops.head(5)
shops.info()
shops.describe()
sales_train.head(5)
sales_train.info()
sales_train.describe()
sales_train.groupby('shop_id').mean() # mean values by shop
sns.set(rc={'figure.figsize':(13,13)})

ax = sns.barplot(x=sales_train.groupby('shop_id').mean().index, y=sales_train.groupby('shop_id').mean()['item_cnt_day'], color="salmon")
sales_train.groupby('shop_id').sum() # sum of values by shop
sub_sales_df = sales_train.groupby('shop_id').sum()

sub_sales_df['index_shop'] = sub_sales_df.index

sub_sales_df = sub_sales_df.sort_values(['item_cnt_day']).reset_index(drop=True)
sns.set(rc={'figure.figsize':(13,13)})

ax = sns.barplot(x=sub_sales_df['index_shop'], y=sub_sales_df['item_cnt_day'], order=sub_sales_df['index_shop'],color="salmon")
sns.set(rc={'figure.figsize':(10,10)})

ax = sns.kdeplot(sales_train['item_price'], color="black", shade=True)
print('Count of prices overall:', len(sales_train))

print('Count of prices < 50000:', len(sales_train[sales_train['item_price'] < 50000]))

print('Count of prices 50000 <= x <= 250000:', len(sales_train) - len(sales_train[sales_train['item_price'] > 250000]) - len(sales_train[sales_train['item_price'] < 50000]))

print('Count of prices > 250000:', len(sales_train[sales_train['item_price'] > 250000]))
sns.set(rc={'figure.figsize':(10,10)})

ax = sns.kdeplot(sales_train['item_cnt_day'], color="green", bw=1.5, shade=True)
print('Count of items overall:', len(sales_train))

print('Count of items < 0:', len(sales_train[sales_train['item_cnt_day'] < 0]))

print('Count of items < 10:', len(sales_train[sales_train['item_cnt_day'] < 10]))

print('Count of items 10 <= x <= 100:', len(sales_train) - len(sales_train[sales_train['item_cnt_day'] > 100]) - len(sales_train[sales_train['item_cnt_day'] < 10]))

print('Count of items > 100:', len(sales_train[sales_train['item_cnt_day'] > 100]))
sns.set(rc={'figure.figsize':(12,10)})

ax = sns.pointplot(sales_train['date'], sales_train['date_block_num'], color="red")

ax.set_xlabel('')
print('Count of time blocks: ', len(sales_train['date_block_num'].unique()))
test.head(5)
test.info()
test.describe()
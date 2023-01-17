from pathlib import Path



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
ROOT = Path('/kaggle/')

INPUT_DIR = ROOT / 'input/competitive-data-science-predict-future-sales'
list(INPUT_DIR.glob('*'))
item_df = pd.read_csv(INPUT_DIR / 'items.csv')

item_df.head()
shop_df = pd.read_csv(INPUT_DIR / 'shops.csv')

shop_df.head()
item_cat_df = pd.read_csv(INPUT_DIR / 'item_categories.csv')

item_cat_df.head()
train_df = pd.read_csv(INPUT_DIR / 'sales_train.csv')

train_df.head()
test_df = pd.read_csv(INPUT_DIR / 'test.csv')

test_df.head()
sample_submission = pd.read_csv(INPUT_DIR / 'sample_submission.csv')

sample_submission.head()
train_df = train_df.join(item_df, on='item_id', how='outer', lsuffix='', rsuffix='_r')[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day', 'item_name', 'item_category_id']]
train_df.head()
train_df = train_df.join(item_cat_df, on='item_category_id', how='outer', lsuffix='', rsuffix='_r')[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day', 'item_name', 'item_category_id', 'item_category_name']]
train_df.head()
train_df = train_df.join(shop_df, on='shop_id', how='outer', lsuffix='', rsuffix='_r')[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day', 'item_name', 'item_category_id', 'item_category_name', 'shop_name']]
train_df.head()
train_df.reset_index(drop=True, inplace=True)
train_df.dropna(inplace=True)

train_df.drop_duplicates(inplace=True)
train_df[['day', 'month', 'year']] = train_df.date.str.split('.', expand=True)

train_df.day = train_df.day.apply(lambda x: int(x))

train_df.month = train_df.month.apply(lambda x: int(x))

train_df.year = train_df.year.apply(lambda x: int(x))

train_df.head()
train_df.describe()
year_count = train_df.groupby('year').count().item_id.reset_index()

year_count.columns = ['year', 'total_bill']

month_count = train_df.groupby('month').count().item_id.reset_index()

month_count.columns = ['month', 'total_bill']

day_count = train_df.groupby('day').count().item_id.reset_index()

day_count.columns = ['day', 'total_bill']



fig, axes = plt.subplots(1, 3, figsize=(20, 4))

sns.barplot(x='year', y='total_bill', data=year_count, ax=axes[0])

sns.barplot(x='month', y='total_bill', data=month_count, ax=axes[1])

sns.barplot(x='day', y='total_bill', data=day_count, ax=axes[2])

plt.show()
date_block_count = train_df.groupby('date_block_num').count().item_id.reset_index()

date_block_count.columns = ['date_block', 'total_bill']

fig = plt.figure(figsize=(12, 4))

ax = fig.add_axes([0, 0, 1, 1])

sns.barplot(x='date_block', y='total_bill', data=date_block_count, ax=ax)

plt.show()
import pandas as pd
import numpy as np
import itertools
import os
from matplotlib import pyplot
import matplotlib as mpl

%matplotlib inline
%load_ext autoreload
%autoreload 2
data_path = os.path.join(os.pardir, 'input')
train = pd.read_csv(os.path.join(data_path, 'sales_train.csv'))
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')

test = pd.read_csv(os.path.join(data_path, 'test.csv'))
shops = pd.read_csv(os.path.join(data_path, 'shops.csv'))
items = pd.read_csv(os.path.join(data_path, 'items.csv'))
categories = pd.read_csv(os.path.join(data_path, 'item_categories.csv'))
print('Train size:', train.shape[0])
print('Test size:', test.shape[0])
print('Number of items:', items.shape[0])
print('Number of shops:', shops.shape[0])
print('Number of categories:', categories.shape[0])
train.nunique()
test.nunique()
fig, ax = pyplot.subplots(1, 2, figsize=(15, 4))
ax[0].hist(train.item_cnt_day, 100, edgecolor='k', alpha=0.5)
ax[0].set_ylim([0, 15])
ax[0].set_xlabel('item_cnt_day')
ax[0].set_ylabel('Number of items')
ax[0].set_title('Items sold')

ax[1].hist(train.item_price, 100, edgecolor='k', alpha=0.5)
ax[1].set_ylim([0, 15])
# ax[1].set_xlim([0, 25000])
ax[1].set_xlabel('item price')
ax[1].set_ylabel('Number of items')
ax[1].set_title('Items price')
pass
date = []
num_shops = []
num_items = []

for idx, group in train.groupby('date_block_num'):
    num_shops.append(len(group.shop_id.unique()))
    num_items.append(len(group.item_id.unique()))
    date.append((group.date.dt.month.iloc[0], group.date.dt.year.iloc[0]))
fig, ax = pyplot.subplots(1, 2, figsize=(16, 4))
x = np.arange(len(date))
ax[0].bar(x, num_shops, edgecolor='k', alpha=0.75)
ax[0].axhline(len(shops), color='#ff7f0e', linewidth=3)
ax[0].set_xticks(x)
ax[0].set_xticklabels(date, rotation='vertical')
ax[0].set_title('Number of shops per month')
ax[0].set_xlabel('Month, year')
ax[0].set_ylabel('Number of shops')

ax[1].bar(x, num_items, edgecolor='k', alpha=0.75)
ax[1].axhline(len(items), color='#ff7f0e', linewidth=3)
ax[1].set_xticks(x)
ax[1].set_xticklabels(date, rotation='vertical')
ax[1].set_title('Number of items per month')
ax[1].set_xlabel('Month, year')
ax[1].set_ylabel('Number of items');
total_sales = (train.groupby('date_block_num')
               .agg({'item_cnt_day': 'sum', 'item_price': 'mean'})
               .rename({'item_cnt_day': 'item_cnt_month', 'item_price': 'mean_price'}, axis=1))
fig, ax = pyplot.subplots(2, 1, sharex=True, figsize=(12, 7))
total_sales.item_cnt_month.plot(ax=ax[0], marker='o')
ax[0].set_ylabel('Total items solds')

total_sales.mean_price.plot(ax=ax[1], marker='o')
ax[1].set_ylabel('Mean items price')

for axi in ax:
    axi.set_xticks(x)
    axi.set_xticklabels(date, rotation='vertical')
    axi.grid(alpha=0.5)

ax[0].set_title('Number of sales in each month')
ax[1].set_title('Mean item price in each month')
pyplot.tight_layout()
test_combs = set(map(tuple, test[['shop_id', 'item_id']].values))

train_not_test = []
test_not_train = []
train_size = []
for idx, group in train.groupby('date_block_num'):   
    train_combs = set(map(tuple, group[['shop_id', 'item_id']].values))
    train_size.append(len(train_combs))
    train_not_test.append(len(train_combs - test_combs))
    test_not_train.append(len(test_combs - train_combs))

fig, ax = pyplot.subplots(2, 1, sharex=True, figsize=(12, 6))
ax[0].plot(x, train_not_test, '-o', label='Not in test')
ax[0].plot(x, train_size, '-o', label='Train size')
ax[0].set_title('Combinations in train')
ax[0].set_ylabel('Number of (shop, item)')
ax[0].legend()

ax[1].plot(x, test_not_train, '-o', label='Not in train')
ax[1].plot(x, [len(test_combs)]*len(x), '-o', color='#ff7f0e', label='Test size')
ax[1].set_title('Combinations in test')
ax[1].set_ylabel('Number of (shop, item)')
ax[1].legend()

for axi in ax:
    axi.set_xticks(x)
    axi.set_xticklabels(date, rotation='vertical')
    axi.grid(alpha=0.5)

pyplot.tight_layout()
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = [] 
for block_num in train['date_block_num'].unique():
    cur_shops = train[train['date_block_num'] == block_num]['shop_id'].unique()
    cur_items = train[train['date_block_num'] == block_num]['item_id'].unique()
    grid.append(np.array(list(itertools.product(cur_shops, cur_items, [block_num])),dtype='int32'))

#turn the grid into pandas dataframe
grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)

#get aggregated values for (shop_id, item_id, month)
gb = train.groupby(index_cols,as_index=False).agg({'item_cnt_day': 'sum'})\
        .rename(columns={'item_cnt_day': 'target'})

#join aggregated data to the grid
all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)
#sort the data
all_data.sort_values(['date_block_num', 'shop_id', 'item_id'],inplace=True)
all_data.head()
train_not_test = []
test_not_train = []
train_size = []
for idx, group in all_data.groupby('date_block_num'):   
    train_combs = set(map(tuple, group[['shop_id', 'item_id']].values))
    train_size.append(len(train_combs))
    train_not_test.append(len(train_combs - test_combs))
    test_not_train.append(len(test_combs - train_combs))
fig, ax = pyplot.subplots(2, 1, sharex=True, figsize=(12, 6))
ax[0].plot(x, train_not_test, '-o', label='Not in test')
ax[0].plot(x, train_size, '-o', label='Train size')
ax[0].set_title('Combinations in train')
ax[0].set_ylabel('Number of (shop, item)')
ax[0].legend()

ax[1].plot(x, test_not_train, '-o', label='Not in train')
ax[1].plot(x, [len(test_combs)]*len(x), '-o', color='#ff7f0e', label='Test size')
ax[1].set_title('Combinations in test')
ax[1].set_ylabel('Number of (shop, item)')
ax[1].legend()

for axi in ax:
    axi.set_xticks(x)
    axi.set_xticklabels(date, rotation='vertical')
    axi.grid(alpha=0.5)

pyplot.tight_layout()
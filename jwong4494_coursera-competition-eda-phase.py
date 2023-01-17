# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from itertools import product

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
#Reading project data

items=pd.read_csv('../input/items.csv')
item_categories=pd.read_csv('../input/item_categories.csv')
shops=pd.read_csv('../input/shops.csv')

test=pd.read_csv('../input/test.csv.gz',compression='gzip')
sample_submission=pd.read_csv('../input/sample_submission.csv.gz',compression='gzip')
train=pd.read_csv('../input/sales_train.csv.gz',compression='gzip')
# Understand How many Items are there
print(items.info())
print(items.describe())
items.head(5)
# To understand a little more about the item categories
print(item_categories.info())
print(item_categories.describe())
item_categories.head(5)
train.head(5)
test.head(5)
# Adding item categories to the data set
items_to_add = ['item_category_id', 'item_name']

for item in items_to_add:
    train[item] = train['item_id'].map(items[item])
    test[item] = test['item_id'].map(items[item])

train['item_category_name'] = (train['item_category_id'].map
                               (item_categories['item_category_name']))
test['item_category_name'] = (test['item_category_id'].map
                               (item_categories['item_category_name']))

# Adding dates 
train['date'] = pd.to_datetime(train['date'], format="%d.%m.%Y")
train['year'], train['day'] = train['date'].dt.year, train['date'].dt.day
train['month'] = train['date'].dt.month
# See How many items are in each price bins
max_item_prices = train.groupby('item_id')['item_price'].max()
min_item_prices = train.groupby('item_id')['item_price'].min()

print('---- MAX PRICES ----\n', max_item_prices.describe())
print('\n---- MIN PRICES ----\n', min_item_prices.describe())

expensive_items = sum(max_item_prices > 10000)
cheap_items = sum(min_item_prices < 0)

print('\nThere are {0} number of items cost more than $10,000.\
        \nThere are {1} number of item cost less than $0'.format(
    expensive_items, cheap_items))
'''
There seems to be an item that costs $307,980.
There is an item that cost -$1. need to investigate more.

I am going to exclude the expensive items in this study.
'''
plt.figure()
plt.hist(max_item_prices[max_item_prices < 10000])
plt.hist(min_item_prices[(0 < min_item_prices) & (min_item_prices < 10000)])
'''
From the above history, we can tell that most items are in the range of
$0 - $500. Therefore, we are going to study more items prices in that
range.
'''

plt.figure()
plt.hist(max_item_prices[max_item_prices < 500])
plt.hist(min_item_prices[(0 < min_item_prices) & (min_item_prices < 500)])
# count how many items sold in each stores
train_store_item_diversity = train.groupby('shop_id')['item_id'].nunique()

print(train_store_item_diversity.describe())
plt.figure()
plt.plot(train_store_item_diversity)
# count how many items sold per item
train_item_sold = train.groupby('item_id')['item_cnt_day'].sum()

print(train_item_sold.describe())

plt.figure()
plt.plot(train_item_sold)
most_bought_item = train[train.item_id == train_item_sold.argmax()]

most_bought_item.item_name.iloc[0]
# which item gets returned the most
returned_counts = -train[train['item_cnt_day'] < 0].groupby('item_id')['item_cnt_day'].sum()

print(returned_counts.describe())

plt.figure()
plt.plot(returned_counts)
most_returned_item = train[train.item_id == returned_counts.idxmax()].item_name.iloc[0]

print('The most returned item is: {0}'.format(most_returned_item))
# Calculate Revenues of the items
train['revenue'] = train['item_price'] * train['item_cnt_day']
items_num = train.groupby(['month', 'day'])['item_cnt_day'].sum()

print(items_num.describe())
# How shopping trends vary in a year
items_num.plot()
# see which item makes the most revenue
item = train.groupby('item_id')['revenue'].sum().idxmax()

print('Item ID {} is the most profitable item '.format(item))

# see which item makes the least revenue
item = train.groupby('item_id')['revenue'].sum().idxmin()

print('Item ID {} is the least profitable item '.format(item))
#There is no null values/missing values
train.isnull().values.sum()
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = [] 
for block_num in train['date_block_num'].unique():
    cur_shops = train[train['date_block_num']==block_num]['shop_id'].unique()
    cur_items = train[train['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

#turn the grid into pandas dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

#get aggregated values for (shop_id, item_id, month)
gb = train.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})

#fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
#join aggregated data to the grid
all_data = pd.merge(grid,gb,how='left',on=index_cols).fillna(0)
#sort the data
all_data.sort_values(['date_block_num','shop_id','item_id'],inplace=True)
# Expanding Mean Encoding
cumsum = all_data.groupby('item_id').target.cumsum() - all_data.target
cumcnt = all_data.groupby('item_id').target.cumcount()

all_data['item_target_enc'] = cumsum / cumcnt

# Fill NaNs
all_data['item_target_enc'].fillna(0.3343, inplace=True) 
# Clean up item category name - using the first part
item_categories['meta_category'] = item_categories.item_category_name.apply(lambda x: x.split()[0])
print('There are {} unique meta categories.'.format(item_categories['meta_category'].nunique()))
sales = pd.read_csv('../input/sales_train.csv.gz',compression='gzip')
sales['item_category_id'] = sales['item_id'].map(items['item_category_id'])
# Create "grid" with columns
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = []
for block_num in sales['date_block_num'].unique():
    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
# cleaning the outliers
sales = sales[sales.item_price<100000]
sales = sales[sales.item_cnt_day<=1000]
mean_sales = sales.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': 'sum','item_price': np.mean}).reset_index()
mean_sales = pd.merge(grid,mean_sales,on=['date_block_num','shop_id','item_id'],how='left').fillna(0)
# adding the category id too
mean_sales = pd.merge(mean_sales,items,on=['item_id'],how='left')
for type_id in ['item_id','shop_id','item_category_id']:
    for column_id,aggregator,aggtype in [('item_price',np.mean,'avg'),('item_cnt_day',np.sum,'sum'),('item_cnt_day',np.mean,'avg')]:

        mean_df = sales.groupby([type_id,'date_block_num']).aggregate(aggregator).reset_index()[[column_id,type_id,'date_block_num']]
        mean_df.columns = [type_id+'_'+aggtype+'_'+column_id,type_id,'date_block_num']

        mean_sales = pd.merge(mean_sales,mean_df,on=['date_block_num',type_id],how='left')
lag_variables  = list(mean_sales.columns[7:])+['item_cnt_day']
lags = [1 ,2 ,3 ,6]
for lag in lags:
    sales_new_df = mean_sales.copy()
    sales_new_df.date_block_num+=lag
    sales_new_df = sales_new_df[['date_block_num','shop_id','item_id']+lag_variables]
    sales_new_df.columns = ['date_block_num','shop_id','item_id']+ [lag_feat+'_lag_'+str(lag) for lag_feat in lag_variables]
    mean_sales = pd.merge(mean_sales, sales_new_df,on=['date_block_num','shop_id','item_id'] ,how='left')
# fill nan with zeros
for feat in mean_sales.columns:
    if 'item_cnt' in feat:
        mean_sales[feat] = mean_sales[feat].fillna(0)
    elif 'item_price' in feat:
        mean_sales[feat] = mean_sales[feat].fillna(mean_sales[feat].median())

# drop non-lag features
cols_to_drop = lag_variables[:-1] + ['item_name','item_price']

# recent
mean_sales = mean_sales[mean_sales['date_block_num'] > 12]

# Split X_train and X_valid
X_train = mean_sales[mean_sales['date_block_num']<33].drop(cols_to_drop, axis=1)
X_valid =  mean_sales[mean_sales['date_block_num']==33].drop(cols_to_drop, axis=1)
# limit the range of max items
def limit_sales(x):
    if x>40:
        return 40
    elif x<0:
        return 0
    else:
        return x
X_train['item_cnt_day'] = X_train.apply(lambda x: limit_sales(x['item_cnt_day']),axis=1)
X_valid['item_cnt_day'] = X_valid.apply(lambda x: limit_sales(x['item_cnt_day']),axis=1)

import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sales = pd.read_csv('../input/sales_train.csv')
items = pd.read_csv('../input/items.csv')
item_categories = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
sales.head()
items.head()
item_categories.head()
shops.head()
test.head()
# Examine the size & shape of the data
print(sales.shape, items.shape, item_categories.shape, shops.shape, test.shape)
transactions = sales.groupby('date_block_num')['date'].count()
sns.set()
transactions.plot.line(title='Number of transactions by month', color='gray')
shop_counts = sales.groupby('date_block_num')['shop_id'].nunique()
item_counts = sales.groupby('date_block_num')['item_id'].nunique()
fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
shop_counts.plot.line(ax=axarr[0], color='gray')
item_counts.plot.line(ax=axarr[1], color='gray')
axarr[0].set_title('Number of shops with transactions')
axarr[1].set_title('Number of items with transactions')
print(sales['item_id'].nunique(), sales['shop_id'].nunique())
test_shops = test['shop_id'].unique()
print(len(test_shops))
print(np.sort(test_shops))
test_items = test['item_id'].unique()
print(len(test_items))
check = sales[np.isin(sales['item_id'], test_items, invert=True)]
print(len(check['item_id'].unique()))
print(np.sort(check['item_id'].unique()))
# Create grid with columns
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items from that month
grid = [] 
for block_num in sales['date_block_num'].unique():
    cur_shops = sales[sales['date_block_num']==block_num]['shop_id'].unique()
    cur_items = sales[sales['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

# Add the shop_id from the test data and create date_block_num 34
block_num = 34
cur_shops = test['shop_id'].unique()
cur_items = test['item_id'].unique()
grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype='int32'))
    
# Turn the grid into pandas dataframe
grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)
len(grid)
# The size of the grid should be the same as the sum of the product of unique `shop_counts` & `item_counts` for
# both the sales & test data
shop_counts = sales.groupby('date_block_num')['shop_id'].nunique()
item_counts = sales.groupby('date_block_num')['item_id'].nunique()
test_shops = test['shop_id'].nunique()
test_items = test['item_id'].nunique()
print(shop_counts.dot(item_counts) + test_shops * test_items)
# Get aggregated values for (shop_id, item_id, month)
gb = sales.groupby(index_cols, as_index=False)['item_cnt_day'].agg('sum')
# Rename column
gb = gb.rename(columns={'item_cnt_day':'target'})
# Join aggregated data to the grid
all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)
all_data.head()
all_data['target'].describe()
all_data['target'].plot.hist(color='gray')
all_data['target'] = np.clip(all_data['target'], 0, 20)
all_data['target'].plot.hist(color='gray')
# Generate output file in csv format
all_data.to_csv('all_data.csv', index=False)

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import gc
gc.enable()
train = pd.read_csv('../input/sales_train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape)
print(test.shape)
train.head()
test.head()
train.isnull().sum()
print('Before drop train shape:', train.shape)
train.drop_duplicates(subset=['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day'], 
                      keep='first', inplace=True)
train.reset_index(drop=True, inplace=True)
print('After drop train shape:', train.shape)
items = pd.read_csv('../input/items.csv')
item_categories = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')
items.isnull().sum()
item_categories.isnull().sum()
shops.isnull().sum()
print(train.item_price.min())
print(train.item_price.max())
print(train.item_price.mean())
print(train.item_price.median())
print(train.item_price.value_counts().sort_index(ascending=False))
# -1 and 307980 looks like outliers, let's delete them
print('before train shape:', train.shape)
train = train[(train.item_price > 0) & (train.item_price < 300000)]
print('after train shape:', train.shape)
(train.item_id < 0).sum()
(train.shop_id < 0).sum()
import matplotlib 
from matplotlib import pyplot as plt
%matplotlib inline
print(train.item_cnt_day.min())
print(train.item_cnt_day.max())
print(train.item_cnt_day.mean())
print(train.item_cnt_day.median())
train.item_cnt_day.hist()
train.groupby('date_block_num').sum()['item_cnt_day'].hist()
plt.title('Sales per month histogram')
plt.plot(train.groupby('date_block_num').sum()['item_cnt_day'])
plt.title('Sales per month')
print(train.item_price.min())
print(train.item_price.max())
print(train.item_price.mean())
print(train.item_price.median())
train.item_price.hist()
train.item_price.map(np.log1p).hist()
train.loc[:,'item_price'] = train.item_price.map(np.log1p)
train.date_block_num.value_counts().sort_index()
train.shop_id.hist()
train.item_id.hist()
train.columns
print('# of date_block_num:', train.date_block_num.nunique())
print('# of shop ids:', train.shop_id.nunique())
print('# of item ids:', train.item_id.nunique())
print('max # of total combinations:', train.date_block_num.nunique()*train.shop_id.nunique()*train.item_id.nunique())
index_cols = ['date_block_num', 'shop_id', 'item_id']
#date_block_nums, shop_ids, item_ids = zip(*train.groupby(index_cols).mean().index.values)
#print(pd.Series(date_block_nums).value_counts())
# For every month we create a grid from all shops/items combinations from that month
from itertools import product
new_train = [] 
for block_num in train['date_block_num'].unique():
    cur_shops = train.loc[train['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = train.loc[train['date_block_num'] == block_num, 'item_id'].unique()
    new_train.append(np.array(list(product(*[[block_num], cur_shops, cur_items])),dtype='int32'))

# Turn the grid into a dataframe
new_train = pd.DataFrame(np.vstack(new_train), columns = index_cols, dtype=np.int32)
print(train.shape)
print(new_train.shape)
# Groupby data to get shop-item-month aggregates
gb_cols = ['date_block_num', 'shop_id', 'item_id']
gb = train.groupby(gb_cols, as_index=False).agg(
    {'item_cnt_day':{'shop_item_target':'sum', 
                     'shop_item_target_std': np.std,
                     'shop_item_trans_days': 'count'},
     'item_price':{'shop_item_price_med':np.median, 
                   'shop_item_price_mean': np.mean,
                   'shop_item_price_std': np.std},
    })

# Fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values] 
# Join it to the grid
new_train = pd.merge(new_train, gb, how='left', on=gb_cols).fillna(0.); del gb
print(new_train.isnull().sum().max())
new_train.head()
new_train.shop_item_target = np.clip(new_train.shop_item_target.values, 0, 20).astype(np.int32)
new_train.shop_item_target.value_counts().sort_index(ascending=True)
# Groupby data to get shop-month aggregates
gb_cols = ['date_block_num', 'shop_id']
gb = train.groupby(gb_cols, as_index=False).agg(
    {'item_cnt_day':{'shop_target':'sum', 
                     'shop_target_med':np.median,
                     'shop_target_mean': np.mean,
                     'shop_target_std': np.std},
     'item_price':{'shop_price_med':np.median, 
                   'shop_price_mean': np.mean,
                   'shop_price_std': np.std},
    })

# Fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values] 
# Join it to the grid
new_train = pd.merge(new_train, gb, how='left', on=gb_cols).fillna(0.); del gb
print(new_train.isnull().sum().max())
new_train.head()
# Groupby data to get item-month aggregates
gb_cols = ['date_block_num', 'item_id']
gb = train.groupby(gb_cols, as_index=False).agg(
    {'item_cnt_day':{'item_target':'sum', 
                     'item_target_med':np.median,
                     'item_target_mean': np.mean,
                     'item_target_std': np.std},
     'item_price':{'item_price_med':np.median, 
                   'item_price_mean': np.mean,
                   'item_price_std': np.std},
    })

# Fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values] 
# Join it to the grid
new_train = pd.merge(new_train, gb, how='left', on=gb_cols).fillna(0.); del gb
print(new_train.isnull().sum().max())
new_train.head()
del train; gc.collect()
train = new_train
def downcast_dtypes(df):
    '''
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast
    copy_cast_cols = df.loc[:,float_cols].astype(np.float32) # astype->copy : bool, default True.
    df.loc[:,float_cols] = copy_cast_cols; del copy_cast_cols; gc.collect()
    
    copy_cast_cols = df.loc[:,int_cols].astype(np.int32)
    df.loc[:,int_cols] = copy_cast_cols; del copy_cast_cols; gc.collect()
    
    del float_cols, int_cols; gc.collect()
downcast_dtypes(train)
gc.collect()
items.head()
item_categories.head()
shops.head()
train = pd.merge(train, items, how='left', on='item_id')
train = pd.merge(train, item_categories, how='left', on='item_category_id')
train = pd.merge(train, shops, how='left', on='shop_id')

test = pd.merge(test, items, how='left', on='item_id')
test = pd.merge(test, item_categories, how='left', on='item_category_id')
test = pd.merge(test, shops, how='left', on='shop_id')

downcast_dtypes(train)
downcast_dtypes(test)
print(train.info())
print(train.isnull())
print(train.isnull().sum().max())
train.head()
print(test.info())
print(test.isnull())
print(test.isnull().sum().max())
test.head()
gc.collect()
test.loc[:,'date_block_num'] = train.date_block_num.max()+1

for col in train.columns.values:
    if col not in test.columns.values:
        test.loc[:,col] = 0
        
test.head()
features = train.columns.values
ntrain = train.shape[0]
ntest = test.shape[0]
merge = pd.concat([train[features], test[features]], ignore_index=True, copy=False)

print(ntrain, ntest, merge.shape)
merge.head()
test_IDs = test['ID'].values
merge.isnull().sum().max()
downcast_dtypes(merge)
merge.info()
shop_cnt_per_month = merge.groupby('date_block_num')['shop_id'].nunique()
cum_shop_per_month = (shop_cnt_per_month.cumsum() - shop_cnt_per_month) / np.arange(0,len(shop_cnt_per_month))
merge.loc[:, 'cum_shop_num'] = merge['date_block_num'].map(cum_shop_per_month.fillna(0.)).fillna(0.).astype(np.float32)

del cum_shop_per_month, shop_cnt_per_month; gc.collect()
item_cnt_per_month = merge.groupby('date_block_num')['item_id'].nunique()
cum_item_cnt_per_month = (item_cnt_per_month.cumsum() - item_cnt_per_month) / np.arange(0,len(item_cnt_per_month))
merge.loc[:, 'cum_item_num'] = merge['date_block_num'].map(cum_item_cnt_per_month.fillna(0.)).fillna(0.).astype(np.float32)

del cum_item_cnt_per_month, item_cnt_per_month; gc.collect()
item_cat_per_month = merge.groupby('date_block_num')['item_category_id'].nunique()
cum_item_cat_per_month = (item_cat_per_month.cumsum() - item_cat_per_month) / np.arange(0,len(item_cat_per_month))
merge.loc[:, 'cum_item_cat'] = merge['date_block_num'].map(cum_item_cat_per_month.fillna(0.)).fillna(0.).astype(np.float32)

del cum_item_cat_per_month, item_cat_per_month; gc.collect()
item_sold_per_month = merge.groupby('date_block_num')['shop_item_target'].sum()
cum_item_saled_per_month = (item_sold_per_month.cumsum() - item_sold_per_month) / np.arange(0,len(item_sold_per_month))
merge.loc[:, 'cum_item_sales'] = merge['date_block_num'].map(cum_item_saled_per_month.fillna(0)).fillna(0.).astype(np.float32)

del cum_item_saled_per_month, item_sold_per_month; gc.collect()
pd.options.display.max_columns=100
merge.head()
from tqdm import tqdm_notebook

merge_grouped = merge.groupby(['shop_id', 'item_id'])
cumcnt = merge_grouped.cumcount()

mean_enc_cols = ['shop_item_target','shop_item_target_std', 'shop_item_trans_days', 
                 'shop_item_price_med', 'shop_item_price_mean', 'shop_item_price_std',
                 'shop_target', 'shop_target_med', 'shop_target_mean', 'shop_target_std',
                 'shop_price_med', 'shop_price_mean', 'shop_price_std', 
                 'item_target', 'item_target_med', 'item_target_mean', 'item_target_std',
                 'item_price_med', 'item_price_mean', 'item_price_std']
mean_encoded_cols = []
for col in tqdm_notebook(mean_enc_cols):
    print('Processing '+col)
    enc_col = 'exp_mean_enc_' + col
    mean_encoded_cols.append(enc_col)
    merge.loc[:, enc_col] = pd.Series((merge_grouped[col].cumsum() - merge[col].values)/cumcnt).fillna(0.0).astype(np.float32)
    gc.collect()
    
    '''
    if 'shop_item_target' == col:
         merge.loc[:, enc_col] = np.clip(merge.loc[:, enc_col].values, 0., 20.)
    '''
del merge_grouped, cumcnt; gc.collect()
merge.head(1000000)
'''
index_cols = ['date_block_num', 'shop_id', 'item_id']
lagged_cols = ['shop_item_target', 'shop_item_price_mean',
               'shop_target', 'shop_price_mean',
               'item_target', 'item_price_mean',
              ]
shift_range = [1,2,3,6,9,12]
print(shift_range)

for month_shift in tqdm_notebook(shift_range):
    
    for lagged_col in lagged_cols:
        merge_shift = merge[index_cols + [lagged_col]].copy()
        
        merge_shift.loc[:,'date_block_num'] = merge_shift['date_block_num'] + month_shift
        
        foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x == lagged_col else x
        merge_shift = merge_shift.rename(columns=foo)
        
        lagged_col_name = '{}_lag_{}'.format(lagged_col, month_shift)
        merge = pd.merge(merge, merge_shift, on=index_cols, how='left')
        merge.loc[:, lagged_col_name] = merge[lagged_col_name].fillna(0.).astype(np.float32)

        del merge_shift; gc.collect()
'''
#merge.head(1000000)
target_col = 'shop_item_target'
drop_columns = mean_enc_cols
del train, test; gc.collect()
train_y = merge[target_col].values

merge.drop(drop_columns, axis=1, inplace=True)
merge.loc[:, 'target'] = train_y
train = merge.loc[0:ntrain-1,:].reset_index(drop=True)
test = merge.loc[ntrain:,:].reset_index(drop=True)

print(train.shape)
print(test.shape)
del merge
gc.collect()
gc.collect()
train.head()
test.head()
print(train.columns.values)
features = train.columns.values
text_features = ['item_name', 'item_category_name', 'shop_name']
features = [f for f in features if f not in text_features]
features.extend(['item_name', 'item_category_name', 'shop_name'])
print(features)
train[features].to_csv('proc_train.csv.gz', index=False, float_format='%.8f', compression='gzip', chunksize=100000)
test[features].to_csv('proc_test.csv.gz', index=False, float_format='%.8f', compression='gzip', chunksize=100000)
test_id_df = pd.DataFrame(data=test_IDs.reshape((len(test_IDs),1)), columns=['ID'])
test_id_df.to_csv('test_id.csv', index=False)

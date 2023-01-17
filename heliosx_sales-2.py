import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
items=pd.read_csv('../input/items.csv')
item_cat=pd.read_csv('../input/item_categories.csv')
shops=pd.read_csv('../input/shops.csv')

train=pd.read_csv('../input/sales_train.csv.gz',compression='gzip')
test=pd.read_csv('../input/test.csv.gz',compression='gzip')

sample_submission=pd.read_csv('../input/sample_submission.csv.gz',compression='gzip')
l_cat = list(item_cat.item_category_name)

l_cat[0] = 'PC Headsets'

for ind in range(1,8):
    l_cat[ind] = 'Accessories'
    
l_cat[8] = 'Tickets'  
l_cat[9] = 'Goods'    

for ind in range(10,18):
    l_cat[ind] = 'Game consoles'

for ind in range(18,32):
    l_cat[ind] = 'Games'

for ind in range(32,37):
    l_cat[ind] = 'Payment cards'

for ind in range(37,43):
    l_cat[ind] = 'Movie'

for ind in range(43,55):
    l_cat[ind] = 'Books'

for ind in range(55,61):
    l_cat[ind] = 'Music'

for ind in range(61,73):
    l_cat[ind] = 'Presents'

for ind in range(73,79):
    l_cat[ind] = 'Programs'

l_cat[79] = 'Service'  
l_cat[80] = 'Service'

l_cat[81] = 'Clean media'  
l_cat[82] = 'Clean media'

l_cat[83] = 'Batteries' 

item_cat['cats'] = l_cat
items = pd.merge(items, item_cat, on=['item_category_id'], how='left')
items = items[['item_id', 'cats']]
train.describe()
train.head()
test.describe()
test.head()
train.shop_id.unique().shape
test.shop_id.unique().shape
set(test.shop_id.unique()) < set(train.shop_id.unique())
train.item_id.unique().shape
test.item_id.unique().shape
set(test.item_id.unique()) < set(train.item_id.unique())
train.item_price.hist() #bins=10
train.item_cnt_day.hist()
train = train[train['item_price'] < 100000]
train = train[train['item_cnt_day'] < 1000]
from itertools import product
index_cols = ['shop_id', 'item_id', 'date_block_num']
grid = []
for block_num in train['date_block_num'].unique():
    cur_shops = train.loc[ train['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = train.loc[ train['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))
    
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
train = train.groupby(['date_block_num','shop_id','item_id']).agg(
    {'item_cnt_day': np.sum, 'item_price': np.mean}).reset_index()
train.rename({'item_cnt_day': 'item_cnt_month'}, axis='columns', inplace=True)
train = pd.merge(grid, train, on=index_cols, how='left')
del grid
items = items[['item_id', 'cats']]
train = pd.merge(train, items, on=['item_id'], how='left')
test = pd.merge(test, items, on=['item_id'], how='left')
for type_ids in [['item_id'], ['shop_id'], ['cats']]:
    for column_id in ['item_price', 'item_cnt_month']:
        mean_df = train[type_ids + [column_id]].groupby(type_ids).agg(np.mean).reset_index()
        mean_df.rename(
            {column_id: "mean_of_"+column_id+"_groupby_"+"_".join(type_ids)},
            axis='columns', inplace=True
        )
        
        train = pd.merge(train, mean_df, on=type_ids, how='left')
        test  = pd.merge(test, mean_df, on=type_ids, how='left')
del mean_df
test['mean_of_item_price_groupby_item_id'] = test['mean_of_item_price_groupby_item_id'].fillna(test['mean_of_item_price_groupby_cats'])
test['mean_of_item_cnt_month_groupby_item_id'] = test['mean_of_item_cnt_month_groupby_item_id'].fillna(test['mean_of_item_cnt_month_groupby_cats'])
for df in train, test:
    for feat in df.columns[4:]:
        if 'item_cnt' in feat:
            df[feat]=df[feat].fillna(0)
        elif 'item_price' in feat:
            df[feat]=df[feat].fillna(df[feat].median())
train['item_cnt_month'] = train['item_cnt_month'].fillna(0)
train_temp = train.copy()
train= train[train['date_block_num']>=12]
features = ['item_cnt_month', 'item_price', 'mean_of_item_price_groupby_item_id',
       'mean_of_item_cnt_month_groupby_item_id',
       'mean_of_item_price_groupby_shop_id',
       'mean_of_item_cnt_month_groupby_shop_id',
       'mean_of_item_price_groupby_cats',
       'mean_of_item_cnt_month_groupby_cats',
       ]
def add_data(df):
    for diff in (1,6,12):
        train_copy = train_temp.copy()
        train_copy['date_block_num'] += diff
        train_copy = train_copy[['date_block_num', 'item_id', 'shop_id'] + features]
        train_copy.rename({
            feat: feat+"_"+str(diff)+'_month_ago' for feat in features
        }, axis=1, inplace=True)
        df = pd.merge(df, train_copy, on=['shop_id', 'item_id', 'date_block_num'], how='left')
    return df
test['date_block_num'] = 34
train = add_data(train)
test = add_data(test) 
test.drop('date_block_num', axis=1, inplace=True)
for df in train, test:
    for feat in train.columns[6:]:
        if 'item_cnt' in feat:
            df[feat]=df[feat].fillna(0)
        elif 'item_price' in feat:
            df[feat]=df[feat].fillna(df[feat].median())
del train_temp
train.head()
train['item_cnt_month'] = train['item_cnt_month'].clip(0, 20)
train_set = train[train['date_block_num']<33]
val_set = train[train['date_block_num']==33].reset_index()
features =train_set.columns[6:].tolist()
X_train = train_set[features]

y_train = train_set['item_cnt_month']

X_val = val_set[features]

y_val = val_set['item_cnt_month']

test = test[['ID'] + features]

X_test = test[features]
del train
import xgboost as xgb
params = {
        'eta': 0.08, 
        'max_depth': 7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'seed': 3,
        'gamma':1,
        'silent': True
    }
watchlist = [
    (xgb.DMatrix(X_train, y_train), 'train'),
    (xgb.DMatrix(X_val, y_val), 'validation')
]
model = xgb.train(params, xgb.DMatrix(X_train, y_train), 500,  watchlist, maximize=False, verbose_eval=5, early_stopping_rounds=5)
pred = model.predict( xgb.DMatrix(X_test), ntree_limit = model.best_ntree_limit)
test['item_cnt_month'] = pred.clip(0, 40)
test[['ID', 'item_cnt_month']].to_csv('xgb_submission.csv', index=False)
test[['ID', 'item_cnt_month']].head()
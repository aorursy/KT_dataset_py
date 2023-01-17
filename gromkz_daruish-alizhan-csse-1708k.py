import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
from tqdm import tqdm_notebook
import xgboost as xgb

DATA_FOLDER = ''
item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
transactions = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
test_data       = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
test_ids = test_data['ID']
test_data.drop(['ID'], axis=1,inplace=True)
test_data['date_block_num'] = 34
print(transactions['item_cnt_day'].min())
print(transactions['item_cnt_day'].max())
(transactions['item_cnt_day'].isnull()).describe() 
plt.scatter(range(transactions.shape[0]), transactions['item_cnt_day'])
transactions.loc[transactions['item_cnt_day'] >= 1000]
transactions = transactions[transactions.item_cnt_day <= 1000]
# Also check items with very high price
transactions['item_price'].describe()
transactions.loc[transactions['item_price'] >= 100000]
transactions = transactions[transactions.item_price<=100000]
## Do we have data from the shops equally?
x = transactions['shop_id'].value_counts()
print(x.min())
print(x.max())
plt.scatter(x.index, x.values)
## lets see how much data of transactions we have for each date_block_num
x = transactions['date_block_num'].value_counts()
print(x.min(), x.max())
plt.scatter(x.index, x.values) 
transactions['date'] = pd.to_datetime(transactions['date'], format="%d.%m.%Y")
transactions['day']   = transactions['date'].dt.day
transactions['month'] = transactions['date'].dt.month
transactions['year']  = transactions['date'].dt.year
transactions.drop(['date'], axis=1,inplace=True)
transactions.drop(['month'], axis=1,inplace=True)
year_month_sales = transactions.groupby(['year', 'date_block_num'])['item_cnt_day'].sum()
year_month_sales
plt.plot(range(12), year_month_sales.values[0:12])
plt.plot(range(12), year_month_sales.values[12:24])
plt.plot(range(10), year_month_sales.values[24:34])
transactions.drop(['year'], axis=1,inplace=True)
transactions.head()
# number of unique values
print(len(transactions['date_block_num'].unique()))
print(len(transactions['shop_id'].unique()))
print(len(transactions['item_id'].unique()))
item_category_mapping = items[['item_id','item_category_id']].drop_duplicates()
transactions = pd.merge(transactions, item_category_mapping, how='left', on='item_id')
item_category_mapping = items[['item_id','item_category_id']].drop_duplicates()
test_data = pd.merge(test_data, item_category_mapping, how='left', on='item_id')
# Create corpus of shop names 
shops_list =list( shops['shop_name'])
print(shops['shop_name'].shape)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase=True)
shop_text_features = vectorizer.fit_transform(shops_list)
print(np.shape(shop_text_features))
# convert sparse matrix to dense matrix
feature_names = vectorizer.get_feature_names()
denselist = shop_text_features.todense().tolist()
shop_text_features = pd.DataFrame(denselist, columns=feature_names)
## columns with no more occurance than one are not useful
x = shop_text_features.astype(bool).sum(axis=0)
x = x[x.values > 2]
l = x.index.tolist()
shop_text_features = shop_text_features[l]
shop_text_features['shop_id'] = shops['shop_id']
print(shop_text_features.shape)
shop_text_features.tail()
# Here we add item shop pairs for each month which did not occur and indicate them with zero
# This is important to do

from itertools import product
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = [] 
for block_num in transactions['date_block_num'].unique():

    cur_shops = transactions[transactions['date_block_num']==block_num]['shop_id'].unique()
    cur_items = transactions[transactions['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

gb = transactions.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': 'sum'}).reset_index()
gb = gb.rename({'item_cnt_day': 'target'}, axis=1)

all_data = pd.merge(grid,gb, on=index_cols, how='left').fillna(0)
all_data.head()
all_data.shape
## monthly aggregate of item price
month_agg_count = transactions.groupby(['date_block_num','shop_id','item_id']).agg({'item_price': np.mean}).reset_index()

## monthly aggregate of item cnt per shop
month_agg_shop= transactions.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_day':'sum'}).reset_index()
month_agg_shop = month_agg_shop.rename({'item_cnt_day': 'shop_cnt_month'}, axis=1)

## monthly aggregate of item cnt per item
month_agg_item = transactions.groupby(['date_block_num', 'item_id']).agg({'item_cnt_day': 'sum'}).reset_index()
month_agg_item = month_agg_item.rename({'item_cnt_day': 'item_cnt_month'}, axis = 1)

## monthly aggreate in terms of item category
month_agg_cat = transactions.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_day': 'sum'}).reset_index()
month_agg_cat = month_agg_cat.rename({'item_cnt_day': 'cat_cnt_month'}, axis = 1)

## item price aggreagate for each shop
month_agg_shop_price = transactions.groupby(['date_block_num', 'shop_id']).agg({'item_price': np.mean}).reset_index()
month_agg_shop_price = month_agg_shop_price.rename({'item_price': 'item_price_shop_month'}, axis=1)

## item price aggregate for each item id
month_agg_item_price = transactions.groupby(['date_block_num', 'item_id']).agg({'item_price': np.mean}).reset_index()
month_agg_item_price = month_agg_item_price.rename({'item_price': 'item_price_item_month'}, axis=1)
## merging with the gird made above

data = pd.merge(all_data, month_agg_count, on=['date_block_num', 'shop_id', 'item_id'], how='left').fillna(0)
data = pd.merge(data, month_agg_shop, on=['date_block_num', 'shop_id'], how='left').fillna(0)
data = pd.merge(data, month_agg_item, on=['date_block_num', 'item_id'], how='left').fillna(0)

data = pd.merge(data, item_category_mapping, how='left', on='item_id')
data = pd.merge(data, month_agg_cat, on=['date_block_num', 'item_category_id'], how='left').fillna(0)

data = pd.merge(data, month_agg_shop_price, on=['date_block_num', 'shop_id'], how='left').fillna(0)
data = pd.merge(data, month_agg_item_price, on=['date_block_num', 'item_id'], how='left').fillna(0)
#It us useful to clip target values to 0 to 20 before traning
data['target'] = data['target'].clip(0, 20)
shift_range = [1, 12]
index_cols = ['date_block_num', 'shop_id', 'item_id']
cols_to_rename = ['target', 'shop_cnt_month', 'item_cnt_month', 'item_price','cat_cnt_month', 'item_price_shop_month', 'item_price_item_month']

for month_shift in tqdm_notebook(shift_range):
    train_shift = data[index_cols + cols_to_rename].copy()
    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
    
    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=foo)
    # train data
    data = pd.merge(data, train_shift, on=index_cols, how='left').fillna(0)
    # test data
    test_data = pd.merge(test_data, train_shift, on=index_cols, how='left').fillna(0)
    
    
    del train_shift
# Add the shop_text_features
data = pd.merge(data, shop_text_features, on=['shop_id'], how='left').fillna(0)
test_data = pd.merge(test_data, shop_text_features, on=['shop_id'], how='left').fillna(0)
data['month'] = (data['date_block_num'])%12 + 1
test_data['month'] = 11
data.loc[(data['shop_id'] == 0) & (data['date_block_num'] == 0) & (data['item_id'] == 32) ]['target']
data.loc[(data['shop_id'] == 0) & (data['date_block_num'] == 1) & (data['item_id'] == 32)]['target_lag_1']
# Take all data after highest lag value
data = data[data['date_block_num'] >= 12] 
print(data.shape)
data.head()
dates = data['date_block_num']

y_train = data['target']
y_train.shape
to_drop_cols = ['date_block_num','target', 'item_price', 'shop_cnt_month', 'item_cnt_month', 'cat_cnt_month','item_price_shop_month', 'item_price_item_month']

data.drop(to_drop_cols, axis=1, inplace=True)
test_data.drop(['date_block_num'], axis=1, inplace=True)

print(len(data.columns), len(test_data.columns))
corr = data.corr()
# plt.matshow(corr)
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
lr = LinearRegression()
lr.fit(data, y_train)

pred_lr = lr.predict(test_data)
D_train = xgb.DMatrix(data, label=y_train)

params = {
  'colsample_bynode': 0.8,
  'learning_rate': 1,
  'max_depth': 7,
  'num_parallel_tree': 100,
  'objective': 'reg:squarederror',
  'subsample': 0.8,
  'tree_method': 'hist'
}

evals_result = {}
rounds = 20 
bst = xgb.train(params, D_train, num_boost_round=rounds)
D_test       = xgb.DMatrix(test_data)
pred_xgb     = bst.predict(D_test)

X_test_level2 = np.c_[pred_lr, pred_xgb] 

# np.save('X_test_level2.npy', X_test_level2)
dates_train_level2 = dates[dates.isin([29, 30, 31, 32, 33])]
y_train_level2     = y_train[dates.isin([29, 30, 31, 32, 33])]
y_train_level2.shape
X_train_level2 = np.zeros([y_train_level2.shape[0], 2])
for cur_block_num in [29, 30, 31, 32, 33]:

  print(cur_block_num)

  #Step #1
  tmp_train_data = data[dates < cur_block_num]
  tmp_test_data  = data[dates == cur_block_num]
  tmp_y_train_data = y_train[dates < cur_block_num]

  #Step #2
  lr.fit(tmp_train_data.values, tmp_y_train_data)
  pred_lr = lr.predict(tmp_test_data.values)

  print("lr finished")

  #Step #3
  D_train = xgb.DMatrix(tmp_train_data, label=tmp_y_train_data)
  D_test = xgb.DMatrix(tmp_test_data)
  bst = xgb.train(params, D_train, num_boost_round=20)
  pred_xgb = bst.predict(D_test)

  print("xgb finished")

  #Step #4
  result = np.c_[pred_lr, pred_xgb] 
  X_train_level2[dates_train_level2 == cur_block_num] = result

  # np.save("X_train_level2_"+ str(cur_block_num), X_train_level2)
x = X_train_level2.shape[0]
plt.subplot(1, 2, 1)
plt.scatter( range(x), X_train_level2[:,0])
plt.subplot(1, 2, 2)
plt.scatter(range(x), X_train_level2[:,1])
lrnew = LinearRegression()
lrnew.fit(X_train_level2, y_train_level2)
train_preds = lrnew.predict(X_train_level2)
print(r2_score(y_train_level2, train_preds))
test_pred = lrnew.predict(X_test_level2)
test_pred = test_pred.clip(0, 20)
pd.Series(test_pred).describe()
sub_df = pd.DataFrame({'ID':test_ids,'item_cnt_month': test_pred })
sub_df.head()
sub_df.to_csv('submission.csv',index=False)
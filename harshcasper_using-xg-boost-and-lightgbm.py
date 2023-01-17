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

from sklearn import ensemble, metrics


parser = lambda date: pd.to_datetime(date, format='%d.%m.%Y')



train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], date_parser=parser)

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

print('train:', train.shape, 'test:', test.shape, 'items:', items.shape, 'item_cats:', item_cats.shape, 'shops:', shops.shape)
test_only = test[~test['item_id'].isin(train['item_id'].unique())]['item_id'].unique()

print('test only items:', len(test_only))
subset = ['date','date_block_num','shop_id','item_id','item_cnt_day']

print(train.duplicated(subset=subset).value_counts())

train.drop_duplicates(subset=subset, inplace=True)
test_shops = test.shop_id.unique()

test_items = test.item_id.unique()

train = train[train.shop_id.isin(test_shops)]

train = train[train.item_id.isin(test_items)]



print('train:', train.shape)
from itertools import product



# create all combinations

block_shop_combi = pd.DataFrame(list(product(np.arange(34), test_shops)), columns=['date_block_num','shop_id'])

shop_item_combi = pd.DataFrame(list(product(test_shops, test_items)), columns=['shop_id','item_id'])

all_combi = pd.merge(block_shop_combi, shop_item_combi, on=['shop_id'], how='inner')

print(len(all_combi), 34 * len(test_shops) * len(test_items))



# group by monthly

train_base = pd.merge(all_combi, train, on=['date_block_num','shop_id','item_id'], how='left')

train_base['item_cnt_day'].fillna(0, inplace=True)

train_grp = train_base.groupby(['date_block_num','shop_id','item_id'])
train_monthly = pd.DataFrame(train_grp.agg({'item_cnt_day':['sum','count']})).reset_index()

train_monthly.columns = ['date_block_num','shop_id','item_id','item_cnt','item_order']

print(train_monthly[['item_cnt','item_order']].describe())

# trim count

train_monthly['item_cnt'].clip(0, 20, inplace=True)



train_monthly.head()
item_grp = item_cats['item_category_name'].apply(lambda x: str(x).split(' ')[0])

item_grp = pd.Categorical(item_grp).codes

item_cats['item_group'] = item_grp

#item_cats = item_cats.join(pd.get_dummies(item_grp, prefix='item_group', drop_first=True))



items = pd.merge(items, item_cats.loc[:,['item_category_id','item_group']], on=['item_category_id'], how='left')



city = shops.shop_name.apply(lambda x: str.replace(x, '!', '')).apply(lambda x: x.split(' ')[0])

shops['city'] = pd.Categorical(city).codes
grp = train_monthly.groupby(['shop_id', 'item_id'])

train_shop = grp.agg({'item_cnt':['mean','median','std'],'item_order':'mean'}).reset_index()

train_shop.columns = ['shop_id','item_id','cnt_mean_shop','cnt_med_shop','cnt_std_shop','order_mean_shop']

print(train_shop[['cnt_mean_shop','cnt_med_shop','cnt_std_shop']].describe())



train_shop.head()
train_cat_monthly = pd.merge(train_monthly, items, on=['item_id'], how='left')

grp = train_cat_monthly.groupby(['shop_id', 'item_group'])

train_shop_cat = grp.agg({'item_cnt':['mean']}).reset_index()

train_shop_cat.columns = ['shop_id','item_group','cnt_mean_shop_cat']

print(train_shop_cat.loc[:,['cnt_mean_shop_cat']].describe())



train_shop_cat.head()
train_prev = train_monthly.copy()

train_prev['date_block_num'] = train_prev['date_block_num'] + 1

train_prev.columns = ['date_block_num','shop_id','item_id','cnt_prev','order_prev']



for i in [2,12]:

    train_prev_n = train_monthly.copy()

    train_prev_n['date_block_num'] = train_prev_n['date_block_num'] + i

    train_prev_n.columns = ['date_block_num','shop_id','item_id','cnt_prev' + str(i),'order_prev' + str(i)]

    train_prev = pd.merge(train_prev, train_prev_n, on=['date_block_num','shop_id','item_id'], how='left')



train_prev.head()
grp = pd.merge(train_prev, items, on=['item_id'], how='left').groupby(['date_block_num','shop_id','item_group'])

train_cat_prev = grp['cnt_prev'].mean().reset_index()

train_cat_prev = train_cat_prev.rename(columns={'cnt_prev':'cnt_prev_cat'})

print(train_cat_prev.loc[:,['cnt_prev_cat']].describe())



train_cat_prev.head()
train_piv = train_monthly.pivot_table(index=['shop_id','item_id'], columns=['date_block_num'], values='item_cnt', aggfunc=np.sum, fill_value=0)

train_piv = train_piv.reset_index()

train_piv.head()
col = np.arange(34)

pivT = train_piv[col].T

ema_s = pivT.ewm(span=12).mean().T

ema_l = pivT.ewm(span=26).mean().T

macd = ema_s - ema_l

sig = macd.ewm(span=9).mean()



ema_list = []

for c in col:

  sub_ema = pd.concat([train_piv.loc[:,['shop_id','item_id']],

      pd.DataFrame(ema_s.loc[:,c]).rename(columns={c:'cnt_ema_s_prev'}),

      pd.DataFrame(ema_l.loc[:,c]).rename(columns={c:'cnt_ema_l_prev'}),

      pd.DataFrame(macd.loc[:,c]).rename(columns={c:'cnt_macd_prev'}),

      pd.DataFrame(sig.loc[:,c]).rename(columns={c:'cnt_sig_prev'})], axis=1)

  sub_ema['date_block_num'] = c + 1

  ema_list.append(sub_ema)

    

train_ema_prev = pd.concat(ema_list)

train_ema_prev.head()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))



train_monthly.groupby(['date_block_num']).sum().reset_index()['item_cnt'].plot(ax=ax[0])

train_cat_monthly = pd.merge(train_monthly, items, on=['item_id'], how='left')

train_cat_monthly.pivot_table(index=['date_block_num'], columns=['item_group'], values='item_cnt', aggfunc=np.sum, fill_value=0).plot(ax=ax[1], legend=False)
train_price = train_grp['item_price'].mean().reset_index()

price = train_price[~train_price['item_price'].isnull()]



# last price by shop,item

last_price = price.drop_duplicates(subset=['shop_id', 'item_id'], keep='last').drop(['date_block_num'], axis=1)



# null price by shop,item

'''

mean_price = price.groupby(['item_id'])['item_price'].mean().reset_index()

result_price = pd.merge(test, mean_price, on=['item_id'], how='left').drop('ID', axis=1)

pred_price_set = result_price[result_price['item_price'].isnull()]

'''

uitem = price['item_id'].unique()

pred_price_set = test[~test['item_id'].isin(uitem)].drop('ID', axis=1)
if len(pred_price_set) > 0:

    train_price_set = pd.merge(price, items, on=['item_id'], how='inner')

    pred_price_set = pd.merge(pred_price_set, items, on=['item_id'], how='inner').drop(['item_name'], axis=1)

    reg = ensemble.ExtraTreesRegressor(n_estimators=25, n_jobs=-1, max_depth=15, random_state=42)

    reg.fit(train_price_set[pred_price_set.columns], train_price_set['item_price'])

    pred_price_set['item_price'] = reg.predict(pred_price_set)



test_price = pd.concat([last_price, pred_price_set], join='inner')

test_price.head()
price_max = price.groupby(['item_id']).max()['item_price'].reset_index()

price_max.rename(columns={'item_price':'item_max_price'}, inplace=True)

price_max.head()
train_price_a = pd.merge(price, price_max, on=['item_id'], how='left')

train_price_a['discount_rate'] = 1 - (train_price_a['item_price'] / train_price_a['item_max_price'])

train_price_a.drop('item_max_price', axis=1, inplace=True)

train_price_a.head()
test_price_a = pd.merge(test_price, price_max, on=['item_id'], how='left')

test_price_a.loc[test_price_a['item_max_price'].isnull(), 'item_max_price'] = test_price_a['item_price']

test_price_a['discount_rate'] = 1 - (test_price_a['item_price'] / test_price_a['item_max_price'])

test_price_a.drop('item_max_price', axis=1, inplace=True)

test_price_a.head()
def mergeFeature(source): 

  d = source

  d = pd.merge(d, items, on=['item_id'], how='left').drop('item_group', axis=1)

  d = pd.merge(d, item_cats, on=['item_category_id'], how='left')

  d = pd.merge(d, shops, on=['shop_id'], how='left')



  d = pd.merge(d, train_shop, on=['shop_id','item_id'], how='left')

  d = pd.merge(d, train_shop_cat, on=['shop_id','item_group'], how='left')

  d = pd.merge(d, train_prev, on=['date_block_num','shop_id','item_id'], how='left')

  d = pd.merge(d, train_cat_prev, on=['date_block_num','shop_id','item_group'], how='left')

  d = pd.merge(d, train_ema_prev, on=['date_block_num','shop_id','item_id'], how='left')

  

  d['month'] = d['date_block_num'] % 12

  days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

  d['days'] = d['month'].map(days).astype(np.int8)

  

  d.drop(['shop_id','shop_name','item_id','item_name','item_category_id','item_category_name','item_group'], axis=1, inplace=True)

  d.fillna(0.0, inplace=True)

  return d
train_set = train_monthly[train_monthly['date_block_num'] >= 12]



train_set = pd.merge(train_set, train_price_a, on=['date_block_num','shop_id','item_id'], how='left')

train_set = mergeFeature(train_set)



train_set = train_set.join(pd.DataFrame(train_set.pop('item_order'))) # move to last column



X_train = train_set.drop(['item_cnt'], axis=1)

#Y_train = train_set['item_cnt']

Y_train = train_set['item_cnt'].clip(0.,20.)

X_train.head()
test_set = test.copy()

test_set['date_block_num'] = 34



test_set = pd.merge(test_set, test_price_a, on=['shop_id','item_id'], how='left')

test_set = mergeFeature(test_set)



test_set['item_order'] = test_set['order_prev']

test_set.loc[test_set['item_order'] == 0, 'item_order'] = 1



X_test = test_set.drop(['ID'], axis=1)

X_test.head()



assert(X_train.columns.isin(X_test.columns).all())
from sklearn import linear_model, preprocessing

from sklearn.model_selection import KFold

import xgboost as xgb

import lightgbm as lgb



params={'learning_rate': 0.05,

        'objective':'rmse',

        'metric':'rmse',

        'num_leaves': 31,

        'verbose': 1,

        'random_state':42,

        'bagging_fraction': 0.7,

        'feature_fraction': 0.7

       }



folds = KFold(n_splits=5, random_state=42)

oof_preds = np.zeros(X_train.shape[0])

sub_preds = np.zeros(X_test.shape[0])



for fold_, (trn_, val_) in enumerate(folds.split(X_train[:1000000], Y_train[:1000000])):

    trn_x, trn_y = X_train.iloc[trn_], Y_train[trn_]

    val_x, val_y = X_train.iloc[val_], Y_train[val_]



#     reg = xgb.XGBRegressor(n_estimators=25, max_depth=12, learning_rate=0.1, subsample=1, colsample_bytree=0.9, random_state=42, eval_metric='rmse')

#     reg.fit(trn_x, trn_y)

    reg = lgb.LGBMRegressor(**params, n_estimators=5000)

    reg.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], eval_metric='rmse', early_stopping_rounds=100, verbose=500)

    oof_preds[val_] = reg.predict(val_x)

    sub_preds += reg.predict(X_test) / folds.n_splits
feature_importance = reg.feature_importances_

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5



plt.figure(figsize=(12,6))

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
pred_cnt = sub_preds





result = pd.DataFrame({

    "ID": test["ID"],

    "item_cnt_month": pred_cnt.clip(0. ,20.)

})

result.to_csv("submission.csv", index=False)
print(len(pred_cnt[pred_cnt > 20]))

result.head(10)
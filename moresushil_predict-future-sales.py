import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

sample_sbumission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
#Perfect Pipeline

import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)



from itertools import product

from sklearn.preprocessing import LabelEncoder



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from xgboost import XGBRegressor

from xgboost import plot_importance



def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



import time

import sys

import gc

import pickle

sys.version_info
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

cats = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

# set index to ID to avoid droping it later

test  = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')
plt.figure(figsize=(10,4))

plt.xlim(-100, 3000)

sns.boxplot(x=train.item_cnt_day)



plt.figure(figsize=(10,4))

plt.xlim(train.item_price.min(), train.item_price.max()*1.1)

sns.boxplot(x=train.item_price)
train = train[train.item_price<40000]

train = train[train.item_cnt_day<300]
median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()

train.loc[train.item_price<0, 'item_price'] = median
# Якутск Орджоникидзе, 56

train.loc[train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

train.loc[train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops = shops[['shop_id','city_code']]



cats['split'] = cats['item_category_name'].str.split('-')

cats['type'] = cats['split'].map(lambda x: x[0].strip())

cats['type_code'] = LabelEncoder().fit_transform(cats['type'])

# if subtype is nan then type

cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])

cats = cats[['item_category_id','type_code', 'subtype_code']]



items.drop(['item_name'], axis=1, inplace=True)
len(list(set(test.item_id) - set(test.item_id).intersection(set(train.item_id)))), len(list(set(test.item_id))), len(test)
ts = time.time()

matrix = []

cols = ['date_block_num','shop_id','item_id']

for i in range(34):

    sales = train[train.date_block_num==i]

    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))

    

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)

matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)

matrix['shop_id'] = matrix['shop_id'].astype(np.int8)

matrix['item_id'] = matrix['item_id'].astype(np.int16)

matrix.sort_values(cols,inplace=True)

time.time() - ts
train['revenue'] = train['item_price'] *  train['item_cnt_day']
ts = time.time()

group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

group.columns = ['item_cnt_month']

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=cols, how='left')

matrix['item_cnt_month'] = (matrix['item_cnt_month']

                                .fillna(0)

                                .clip(0,20) # NB clip target here

                                .astype(np.float16))

time.time() - ts
test['date_block_num'] = 34

test['date_block_num'] = test['date_block_num'].astype(np.int8)

test['shop_id'] = test['shop_id'].astype(np.int8)

test['item_id'] = test['item_id'].astype(np.int16)
ts = time.time()

matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)

matrix.fillna(0, inplace=True) # 34 month

time.time() - ts
ts = time.time()

matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')

matrix = pd.merge(matrix, items, on=['item_id'], how='left')

matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')

matrix['city_code'] = matrix['city_code'].astype(np.int8)

matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)

matrix['type_code'] = matrix['type_code'].astype(np.int8)

matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)

time.time() - ts
def lag_feature(df, lags, col):

    tmp = df[['date_block_num','shop_id','item_id',col]]

    for i in lags:

        shifted = tmp.copy()

        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]

        shifted['date_block_num'] += i

        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')

    return df
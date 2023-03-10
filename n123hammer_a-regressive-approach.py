# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
train.head()
shops.head()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))

plt.xlim(-100, 3000)

sns.boxplot(x= train.item_cnt_day)



plt.figure(figsize=(10,4))

plt.xlim(train.item_price.min(), train.item_price.max()*1.1)

sns.boxplot(x=train.item_price)
train = train[train.item_price<100000]

train = train[train.item_cnt_day < 1001]
train.item_price.median()
train[train.item_price < 0]
median_value = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)&(train.item_cnt_day==1)].item_price.median()

median_value
train.loc[train.item_price<0, 'item_price'] = median_value
train[train.shop_id==0].head()
shops.head()
items.head()
train.loc[train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57



train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58



train.loc[train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
item_categories.head()
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops.loc[shops.city == '!????????????', 'city'] = '????????????'

shops



item_categories['split'] = item_categories['item_category_name'].str.split('-')

item_categories['type'] = item_categories['item_category_name'].str.split('-').map(lambda x: x[0])

item_categories['sub_type'] = item_categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())



item_categories.head()
test.head()
matrix = []

from itertools import product

cols = ['date_block_num', 'shop_id', 'item_id']

for i in range(34):

    sales = train[train.date_block_num == i]

    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype = 'int16'))

    

matrix = pd.DataFrame(np.vstack(matrix), columns = cols)

matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)

matrix['shop_id'] = matrix['shop_id'].astype(np.int8)

matrix['item_id'] = matrix['item_id'].astype(np.int16)

matrix.sort_values(cols,inplace=True)

matrix.head()
train.head()
train['revenue'] = train['item_price'] * train['item_cnt_day']



group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

group.columns = ['item_cnt_month']
group.head()
group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=cols, how='left')

matrix['item_cnt_month'] = (matrix['item_cnt_month'].fillna(0).clip(0,20).astype(np.float16))
matrix.head()
test.head()
test['date_block_num'] = 34

test['date_block_num'] = test['date_block_num'].astype(np.int8)

test['shop_id'] = test['shop_id'].astype(np.int8)

test['item_id'] = test['item_id'].astype(np.int16)
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)

matrix.fillna(0, inplace=True)
matrix.drop(columns = ['ID'], inplace = True)

matrix.head()
shops.head()
from sklearn.preprocessing import LabelEncoder

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops = shops[['shop_id','city_code']]
item_categories['type_code'] = LabelEncoder().fit_transform(item_categories['type'])

item_categories['sub_type_code'] = LabelEncoder().fit_transform(item_categories['sub_type'])

item_categories = item_categories[['item_category_id', 'type_code', 'sub_type_code']]

item_categories.head()
items.drop(columns = ['item_name'], inplace = True)

items.head()
matrix = pd.merge(matrix, shops, on = ['shop_id'], how = 'left')

matrix = pd.merge(matrix, items, on = ['item_id'], how = 'left')

matrix = pd.merge(matrix, item_categories, on = ['item_category_id'], how = 'left')

matrix['city_code'] = matrix['city_code'].astype(np.int8)

matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)

matrix['type_code'] = matrix['type_code'].astype(np.int8)

matrix['sub_type_code'] = matrix['sub_type_code'].astype(np.int8)

matrix.head()
def lag_features(df, lags, col):

    tmp = df[['date_block_num', 'shop_id', 'item_id', col]]

    

    for i in lags:

        shifted = tmp.copy()

        shifted.columns = ['date_block_num', 'shop_id', 'item_id', col+'_lag_'+str(i)]

        shifted['date_block_num'] += 1

        

        df = pd.merge(df, shifted, on = ['date_block_num','shop_id','item_id'], how='left')

    return df
matrix = lag_features(matrix, [1,2,3,6,12], 'item_cnt_month')

matrix.head()
group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_avg_item_cnt' ]

group.reset_index(inplace=True)
matrix.head()
matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')

matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)

matrix = lag_features(matrix, [1], 'date_avg_item_cnt')

matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month':['mean']})

group.columns = ['date_item_avg_item_cnt']

group.reset_index(inplace = True)



matrix = pd.merge(matrix, group, on = ['date_block_num', 'item_id'], how = 'left')

matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)

matrix = lag_features(matrix, [1], 'date_item_avg_item_cnt')

matrix.drop(['date_item_avg_item_cnt'], axis = 1, inplace = True)
group = train.groupby(['item_id']).agg({'item_price': ['mean']})

group.columns = ['item_avg_item_price']

group.reset_index(inplace = True)
matrix = pd.merge(matrix, group, on = ['item_id'], how = 'left')

matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)
group = train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})

group.columns = ['date_item_avg_item_price']

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')

matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)
lags = [1,2,3,4,5,6]

matrix = lag_features(matrix, lags, 'date_item_avg_item_price')
for i in lags:

    matrix['delta_price_lag_'+str(i)] = (matrix['date_item_avg_item_price_lag_'+str(i)] - matrix['item_avg_item_price'])/matrix['item_avg_item_price']
def select_trend(row):

    for i in lags:

        if row['delta_price_lag_'+str(i)]:

            return row['delta_price_lag_'+str(i)]

    return 0



matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)

matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)

matrix['delta_price_lag'].fillna(0, inplace=True)



fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']

for i in lags:

    fetures_to_drop += ['date_item_avg_item_price_lag_'+str(i)]

    fetures_to_drop += ['delta_price_lag_'+str(i)]



matrix.drop(fetures_to_drop, axis=1, inplace=True)
matrix.head()
matrix = matrix[matrix.date_block_num > 11]
matrix.reset_index(inplace = True)
matrix.drop(['index'], axis = 1, inplace = True)

matrix.head()
matrix.isnull().sum()
X_train = matrix[matrix.date_block_num < 33].drop(['item_cnt_month'], axis = 1)

Y_train = matrix[matrix.date_block_num < 33]['item_cnt_month']

X_Valid = matrix[matrix.date_block_num ==33].drop(['item_cnt_month'], axis = 1)

Y_Valid = matrix[matrix.date_block_num ==33]['item_cnt_month']

X_test = matrix[matrix.date_block_num == 34].drop(['item_cnt_month'], axis = 1)
X_train.isnull().sum()
X_Valid.isnull().sum()
X_test.isnull().sum()
X_train.fillna(0, inplace = True)

X_Valid.fillna(0, inplace = True)

X_test.fillna(0, inplace = True)

from xgboost import XGBRegressor

model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)



model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_Valid, Y_Valid)], 

    verbose=True, 

    early_stopping_rounds = 10)
Y_pred = model.predict(X_Valid).clip(0, 20)

Y_test = model.predict(X_test).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": Y_test

})

submission.to_csv('xgb_submission.csv', index=False)
X_train.head()
X_train.shape
from numpy import array

def split_sequence(sequence, n_steps):

	X, y = list(), list()

	for i in range(len(sequence)):

		# find the end of this pattern

		end_ix = i + n_steps

		# check if we are beyond the sequence

		if end_ix > len(sequence)-1:

			break

		# gather input and output parts of the pattern

		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

		X.append(seq_x)

		y.append(seq_y)

	return array(X), array(y)
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

n_steps = 3

# split into samples

X, y = split_sequence(raw_seq, n_steps)

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))
X_train_numpy = np.array(X_train)
X_train_numpy = X_train_numpy.reshape((X_train_numpy.shape[0], X_train_numpy.shape[1], n_features))
Y_train_numpy = np.array(Y_train)
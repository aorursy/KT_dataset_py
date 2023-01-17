# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_sub_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')

shops_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

sales_train_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

item_cat_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

test_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

items_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
sales_train_df.head()
shops_df.head()
item_cat_df.head()
items_df.head()
test_df.head()
sample_sub_df.head()
print(sales_train_df.shape)

print(shops_df.shape)

print(item_cat_df.shape)

print(items_df.shape)
# Merge all the given dataframe to a single dataframe

train_df = pd.merge(sales_train_df, shops_df, on='shop_id')

train_df = pd.merge(train_df, items_df, on='item_id')

train_df = pd.merge(train_df, item_cat_df, on='item_category_id')

# train_df.head()
# train_df[(train_df['item_id'] == 22154) & (train_df['item_category_id'] == 37) & (train_df['shop_id'] == 25)]
train_df.groupby('item_id').mean()['item_price']
train_df.shape
# Merge dataframes to get test data features

test_df = pd.merge(test_df, items_df, on='item_id')

test_df = pd.merge(test_df, item_cat_df, on='item_category_id')

# test_df = pd.merge(test_df, train_df.groupby('item_id').mean()['item_price'], how='left', on=['item_id'])

test_df.head()
test_df.shape
train_df.isna().sum()
test_df.isna().sum()
print(f'Shape of training data: {train_df.shape}')

print(f'Shape of test data: {test_df.shape}')
train_df = train_df.drop(['shop_name', 'item_name', 'item_category_name'], axis=1)

test_df = test_df.drop(['item_name', 'item_category_name'], axis=1)
train_df.head()
test_df.sort_values('ID').head()
X_train = train_df.drop(['item_cnt_day'], axis=1)

Y_train = train_df['item_cnt_day']
test_df['date_block_num'] = np.uint8(34)

test_df['month'] = np.uint8(11)

test_df['year'] = np.uint8(223)
X_train['date'] = pd.to_datetime(X_train['date'])

X_train['month'] = np.uint8(X_train['date'].dt.month)

X_train['year'] = np.uint8(X_train['date'].dt.year)
X_train.head()
test_df.head()
# TODO

categoricals = ['item_category_id', 'item_id', 'shop_id', 'date_block_num', 'month', 'year']

non_cat = ['item_price']
params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'rmse'},

            'subsample': 0.4,

            'subsample_freq': 1,

            'learning_rate': 0.2,

            'num_leaves': 25,

            'feature_fraction': 0.8,

            'lambda_l1': 1,

            'lambda_l2': 1

        }



folds = 10

seed = 666



kf = StratifiedKFold(n_splits=folds, shuffle=False, random_state=seed)



models = []

for train_index, val_index in kf.split(X_train, Y_train):

    x_train = X_train[categoricals].iloc[train_index]

    x_val = X_train[categoricals].iloc[val_index]

    y_train = Y_train.iloc[train_index]

    y_val = Y_train.iloc[val_index]

    

    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categoricals)

    lgb_eval = lgb.Dataset(x_val, y_val, categorical_feature=categoricals)

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=500,

                valid_sets=(lgb_train, lgb_eval),

                early_stopping_rounds=100,

                verbose_eval = 100)

    models.append(gbm)
sum([model.predict(test_df.sort_values('ID').drop('ID', axis=1)) for model in models])/folds
sample_sub_df['item_cnt_month'] = sum([model.predict(test_df.sort_values('ID').drop('ID', axis=1)) for model in models])/folds
sample_sub_df['item_cnt_month'] = np.clip(sample_sub_df['item_cnt_month'], 0 ,20)
sample_sub_df.to_csv('submission.csv', index=False)

sample_sub_df
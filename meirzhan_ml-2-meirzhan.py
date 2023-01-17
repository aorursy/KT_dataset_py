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
from datetime import datetime



# Import specific packages

import re

from collections import Counter

from scipy.sparse import csr_matrix

from itertools import compress

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.utils import shuffle

from sklearn.preprocessing import LabelEncoder



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12
item_cat = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

sample = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
# let's join the items and item_cat together.

items = pd.merge(items, item_cat, how='left', on=['item_category_id'])
# Let convert a collection of raw documents to a matrix of TF-IDF features using the class sklearn.feature_extraction.text.TfidfVectorizer

feature_cnt = 25

tfidf = TfidfVectorizer(max_features=feature_cnt)

item_name = pd.DataFrame(tfidf.fit_transform(items['item_name']).toarray())
feature_cnt = 25

tfidf = TfidfVectorizer(max_features=feature_cnt)

item_name = pd.DataFrame(tfidf.fit_transform(items['item_name']).toarray())
def merge_dataframe(df_left, df_right, column_name_prefix):

    for column in df_right.columns.values:

        col = column_name_prefix + str(column)

        df_left[col] = df_right[column]
merge_dataframe(items, item_name, 'item_name')
feature_cnt = 25

tfidf = TfidfVectorizer(max_features=feature_cnt)

item_cat_name = pd.DataFrame(tfidf.fit_transform(items['item_category_name']).toarray())



merge_dataframe(items, item_cat_name, 'item_cat_name')



tfidf = TfidfVectorizer(max_features=feature_cnt)

shop_name = pd.DataFrame(tfidf.fit_transform(shops['shop_name']).toarray())



merge_dataframe(shops, shop_name, 'shop_name')
# filtering records

sales_train = sales_train[(sales_train['item_price']>0) & (sales_train['item_cnt_day']>0)]


item_price_latest = sales_train.sort_values(by=['date'], ascending=False).groupby(['item_id', 'shop_id'], as_index=False)['item_price'].first()
item_price_latest
sales_train['date'] = sales_train['date'].apply(lambda x: datetime.strptime(x, '%d.%m.%Y'))

sales_train['year'] = sales_train['date'].apply(lambda x: x.year)

sales_train['month'] = sales_train['date'].apply(lambda x: x.month)


sales_train1 = sales_train.groupby(['shop_id', 'date_block_num', 'item_id', 'year', 'month'], as_index=False)['item_cnt_day'].sum().rename(columns={'item_cnt_day':'item_cnt_month'})
sales_train1.describe()
# The model can be influenced by the outliers

sales_train1['item_cnt_month'] = sales_train1['item_cnt_month'].clip(0, 20)
# sort sales_train1 with ['date_block_num']

sales_train1 = sales_train1.sort_values('date_block_num')

sales_train1['item_cnt_prev_month'] = sales_train1.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(1).fillna(0)
# for the testing, the item_cnt_prev_month is the last month on the sales_train1

test_item_cnt_prev_month = sales_train1[['item_id', 'shop_id', 'item_cnt_month', 'date_block_num']].sort_values('date_block_num', ascending=False).groupby(['item_id', 'shop_id'], as_index=False).first()

test_item_cnt_prev_month = test_item_cnt_prev_month[['item_id', 'shop_id', 'item_cnt_month']].rename(columns={'item_cnt_month': 'item_cnt_prev_month'})
item_month_mean = sales_train1[sales_train1['date_block_num']!=33].groupby(['item_id', 'shop_id'], as_index=False)['item_cnt_month'].mean().rename(columns={'item_cnt_month': 'item_cnt_month_mean'})

sales_train1 = sales_train1.merge(item_month_mean, how='left', on=['item_id', 'shop_id'])

sales_train1.head()
test_item_month_mean = sales_train1.groupby(['item_id', 'shop_id'], as_index=False)['item_cnt_month'].mean().rename(columns={'item_cnt_month': 'item_cnt_month_mean'})

test_item_month_mean.head()
item_price_avg = sales_train.groupby(['item_id', 'shop_id', 'year', 'month'], as_index=False)['item_price'].mean()

item_price_avg.head()
sales_train2 = pd.merge(sales_train1, item_price_avg, how='left', on=['shop_id','item_id','year','month'])

sales_train3 = pd.merge(sales_train2, items, how='left', on=['item_id'])

sales_train4 = pd.merge(sales_train3, shops, how='left', on=['shop_id'])

train = sales_train4
test2 = pd.merge(test, item_price_latest, how='left', on=['shop_id','item_id'])

test3 = pd.merge(test2, items, how='left', on=['item_id'])

test4 = pd.merge(test3, shops, how='left', on=['shop_id'])

test5 = pd.merge(test4, test_item_cnt_prev_month[['item_id', 'shop_id', 'item_cnt_prev_month']], how='left', on=['item_id', 'shop_id'])

test6 = pd.merge(test5, test_item_month_mean, how='left', on=['item_id', 'shop_id'])

df_test = test6
for col in ['shop_id', 'item_id', 'item_category_id']:

    train[col] = train[col].astype(str)

    df_test[col] = df_test[col].astype(str)
train.sort_values(by=['year','month'], ascending=[False, False]).head()
df_test['year'] = 2015

df_test['month'] = 11

df_test['date_block_num'] = 34
train = shuffle(train, random_state=42)
X = train[[col for col in train.columns.values if col not in ['item_name', 'item_category_name', 'shop_name', 'item_cnt_month', 'item_cnt_prev_month', 'item_cnt_month_mean']]].fillna(0)



y = train['item_cnt_month'].fillna(0)



list_training = list(X['date_block_num']<33)

list_testing = list(X['date_block_num']==33)



X_train2 = X[X['date_block_num']<33]

y_train2 = y[list_training].fillna(0)

X_test2 = X[X['date_block_num']==33]

y_test2 = y[list_testing].fillna(0)
reg = ExtraTreesRegressor(n_estimators=25, n_jobs=-1, max_depth=15, random_state=42)

reg.fit(X_train2, y_train2)



y_train_pred = reg.predict(X_train2)

y_test_pred = reg.predict(X_test2)



rmse_train = np.sqrt(mean_squared_error(y_train2, y_train_pred))

rmse_test = np.sqrt(mean_squared_error(y_test2, y_test_pred))



print('rmse_train:', rmse_train)

print('rmse_test:', rmse_test)
rf_reg2 = RandomForestRegressor(n_estimators=25, max_depth=10, random_state=42)

rf_reg2.fit(X_train2, y_train2)



y_train2_pred = rf_reg2.predict(X_train2)

y_test2_pred = rf_reg2.predict(X_test2)
rmse_train2 = np.sqrt(mean_squared_error(y_train2, y_train2_pred))

rmse_test2 = np.sqrt(mean_squared_error(y_test2, y_test2_pred))



print('rmse_train:', rmse_train2)

print('rmse_test:', rmse_test2)
df_test2 = df_test[[col for col in X_train2.columns.values]].fillna(0)

test_pred = reg.predict(df_test2)


test_pred = np.round(test_pred,1)
test['item_cnt_month'] = test_pred
test.to_csv(path_or_buf='sample_submission.csv', columns=['ID', 'item_cnt_month'], index=False)
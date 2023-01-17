import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly_express as px

import matplotlib.image as mpimg

from tabulate import tabulate

import missingno as msno 

from IPython.display import display_html

from PIL import Image

import gc

import cv2

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

item_cat = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

shop = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
test.head(5)
item_cat.head(5)
items.head(5)
train.head(5)
shop.head(5)
train.shape
test.shape
f = plt.figure(figsize=(12,6))

sns.countplot('date_block_num', data=train)
colors = ["#0101DF", "#DF0101"]



f = plt.figure(figsize=(12,6))

sns.countplot('shop_id', data=train, palette=colors) 

train['item_id'].value_counts(ascending=False)[:10]
items.loc[items['item_id']==20949] 
test.loc[test['item_id']==20949].head(5)
train['item_cnt_day'].sort_values(ascending=False)[:10]
train[train['item_cnt_day'] == 2169]
train[train['item_cnt_day'] == 1000]
items[items['item_id'] == 11373]
train[train['item_id'] == 11373].median()
train = train[train['item_cnt_day'] < 2000]
train['item_price'].sort_values(ascending=False)[:10]
train[train['item_price'] == 307980]
items[items['item_id'] == 6066]
train = train[train['item_price'] < 300000]
train['item_price'].sort_values()[:5]
train[train['item_price'] == -1]
items[items['item_id'] == 2973]
#Lets see if there are other observations for this item (2973) which can help us determine its price

train[train['item_id'] == 2973].head(5)
train[train['item_id'] == 2973].median()
price_correction = train[(train['shop_id'] == 32) & (train['item_id'] == 2973) & (train['date_block_num'] == 4) & (train['item_price'] > 0)].item_price.median()

train.loc[train['item_price'] < 0, 'item_price'] = price_correction
train['item_price'].sort_values(ascending=False)
f = plt.figure(figsize=(12,6))

sns.countplot('shop_id', data=test)
test['item_id'].value_counts(ascending=False)[:5]
shop_train = train['shop_id'].nunique()

shop_test = test['shop_id'].nunique()
shop_train
shop_test
#However, this doesn't mean that the training set contains all of the shops present in the test set.

#For that, we need to see if every element of the test set is present in the training set.

#Let's write some simple code to see if the test set list is a subset of the training set list.

shops_train_list = list(train['shop_id'].unique())

shops_test_list = list(test['shop_id'].unique())



flag = 0

if(set(shops_test_list).issubset(set(shops_train_list))): 

    flag = 1

      

if (flag) : 

    print ("Yes, list is subset of other.") 

else : 

    print ("No, list is not subset of other.") 
train.loc[train['shop_id'] == 0, 'shop_id'] = 57

test.loc[test['shop_id'] == 0, 'shop_id'] = 57



train.loc[train['shop_id'] == 1, 'shop_id'] = 58

test.loc[test['shop_id'] == 1, 'shop_id'] = 58



train.loc[train['shop_id'] == 10, 'shop_id'] = 11

test.loc[test['shop_id'] == 10, 'shop_id'] = 11
cities = shop['shop_name'].str.split(' ').map(lambda row: row[0])
shop['city'] = shop['shop_name'].str.split(' ').map(lambda row: row[0])

shop.loc[shop.city == '!Якутск', 'city'] = 'Якутск'
shop.head(5)
from sklearn import preprocessing

pr = preprocessing.LabelEncoder()

pr.fit_transform(shop['city'])
shop['cities_label'] = pr.fit_transform(shop['city'])

shop.drop(['shop_name', 'city'], axis = 1, inplace=True)
shop.head(5)
item_train = train['item_id'].nunique()

item_test = test['item_id'].nunique()
item_train
item_test
item_train_list = list(train['item_id'].unique())

item_test_list = list(test['item_id'].unique())



flag = 0

if(set(item_test_list).issubset(set(item_train_list))): 

    flag = 1

      

if (flag) : 

    print ("Yes, list is subset of other.") 

else : 

    print ("No, list is not subset of other.") 
len(set(item_test_list).difference(item_train_list))
items_in_test = items.loc[items['item_id'].isin(sorted(test['item_id'].unique()))].item_category_id.unique()
items.loc[~items['item_category_id'].isin(items_in_test)].T
le = preprocessing.LabelEncoder()



main_items = item_cat['item_category_name'].str.split('-')

item_cat['main_category_id'] = main_items.map(lambda row: row[0].strip())

item_cat['main_category_id'] = le.fit_transform(item_cat['main_category_id'])
item_cat['sub_category_id'] = main_items.map(lambda row: row[1].strip() if len(row) > 1 else row[0].strip())

item_cat['sub_category_id'] = le.fit_transform(item_cat['sub_category_id'])
item_cat.head(5)
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
from itertools import product
# Testing generation of cartesian product for the month of January in 2013

shops_in_jan = train.loc[train['date_block_num']==0, 'shop_id'].unique()

items_in_jan = train.loc[train['date_block_num']==0, 'item_id'].unique()
jan = list(product(*[shops_in_jan, items_in_jan, [0]]))
print(len(jan))

shops_in_feb = train.loc[train['date_block_num']==1, 'shop_id'].unique()

items_in_feb = train.loc[train['date_block_num']==1, 'item_id'].unique()

feb = list(product(*[shops_in_feb, items_in_feb, [1]]))
cartesian_test = []

cartesian_test.append(np.array(jan))

cartesian_test.append(np.array(feb))
cartesian_test = np.vstack(cartesian_test)
cartesian_test_df = pd.DataFrame(cartesian_test, columns = ['shop_id', 'item_id', 'date_block_num'])
cartesian_test_df.head(5)
cartesian_test_df.shape
from tqdm import tqdm_notebook



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

    df[float_cols] = df[float_cols].astype(np.float16)

    df[int_cols]   = df[int_cols].astype(np.int16)

    

    return df
months = train['date_block_num'].unique()
cartesian = []

for month in months:

    shops_in_month = train.loc[train['date_block_num'] == month, 'shop_id'].unique()

    items_in_month = train.loc[train['date_block_num'] == month, 'item_id'].unique()

    cartesian.append(np.array(list(product(*[shops_in_month, items_in_month, [month]])), dtype='int32'))
cartesian_df = pd.DataFrame(np.vstack(cartesian), columns = ['shop_id', 'item_id', 'date_block_num'], dtype=np.int32)
cartesian_df.shape
train.head(5)
cartesian_df.head(5)
x = train.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()
x.head(5)
x.shape
new_train = pd.merge(cartesian_df, x, on=['shop_id', 'item_id', 'date_block_num'], how='left').fillna(0)
new_train.head(5)
new_train['item_cnt_month'] = np.clip(new_train['item_cnt_month'], 0, 20)
new_train.sort_values(['date_block_num','shop_id','item_id'], inplace = True)

new_train.head()
#First, let's insert the date_block_num feature for the test set! 

#Using insert method of pandas to place this new column at a specific index.

#This will allow us to concatenate the test set easily to the training set

#before we generate mean encodings and lag features

test.insert(loc=3, column='date_block_num', value=34)
test['item_cnt_month'] = 0
test.head(5)
new_train = new_train.append(test.drop(['ID'], axis=1))
new_train.head(5)
new_train = pd.merge(new_train, shop, on=['shop_id'], how='left')

new_train.head()
new_train = pd.merge(new_train, items.drop('item_name', axis = 1), on=['item_id'], how='left')

new_train.head()
new_train = pd.merge(new_train, item_cat.drop('item_category_name', axis = 1), on=['item_category_id'], how='left')

new_train.head()
def generate_lag(train, months, lag_column):

    for month in months:

        # Speed up by grabbing only the useful bits

        train_shift = train[['date_block_num', 'shop_id', 'item_id', lag_column]].copy()

        train_shift.columns = ['date_block_num', 'shop_id', 'item_id', lag_column+'_lag_'+ str(month)]

        train_shift['date_block_num'] += month

        train = pd.merge(train, train_shift, on=['date_block_num', 'shop_id', 'item_id'], how='left')

    return train
new_train = downcast_dtypes(new_train)
import gc

gc.collect()

%%time

new_train = generate_lag(new_train, [1, 2, 3, 4, 5, 6, 12], 'item_cnt_month')
%%time

group = new_train.groupby(['date_block_num', 'item_id'])['item_cnt_month'].mean().rename('item_month_mean').reset_index()

new_train = pd.merge(new_train, group, on=['date_block_num', 'item_id'], how='left')

new_train = generate_lag(new_train, [1,2,3,4,5,6,12], 'item_month_mean')

new_train.drop(['item_month_mean'], axis=1, inplace=True)
%%time

group = new_train.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].mean().rename('shop_month_mean').reset_index()

new_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id'], how='left')

new_train = generate_lag(new_train, [1,2,3,6,12], 'shop_month_mean')

new_train.drop(['shop_month_mean'], axis=1, inplace=True)
%%time

group = new_train.groupby(['date_block_num', 'shop_id', 'item_category_id'])['item_cnt_month'].mean().rename('item_category_month_mean').reset_index()

new_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')

new_train = generate_lag(new_train, [1, 2], 'item_category_month_mean')

new_train.drop(['item_category_month_mean'], axis=1, inplace=True)
%%time

group = new_train.groupby(['date_block_num', 'main_category_id'])['item_cnt_month'].mean().rename('main_category_month_mean').reset_index()

new_train = pd.merge(new_train, group, on=['date_block_num', 'main_category_id'], how='left')



new_train = generate_lag(new_train, [1], 'main_category_month_mean')

new_train.drop(['main_category_month_mean'], axis=1, inplace=True)
%%time

group = new_train.groupby(['date_block_num', 'sub_category_id'])['item_cnt_month'].mean().rename('sub_category_month_mean').reset_index()

new_train = pd.merge(new_train, group, on=['date_block_num', 'sub_category_id'], how='left')



new_train = generate_lag(new_train, [1], 'sub_category_month_mean')

new_train.drop(['sub_category_month_mean'], axis=1, inplace=True)
new_train = downcast_dtypes(new_train)
import xgboost as xgb
new_train = new_train[new_train.date_block_num > 11]
new_train
import gc

gc.collect()
def fill_na(df):

    for col in df.columns:

        if ('_lag_' in col) & (df[col].isnull().any()):

            df[col].fillna(0, inplace=True)

    return df
new_train = fill_na(new_train)

def xgtrain():

    regressor = xgb.XGBRegressor(n_estimators = 5000,

                                 learning_rate = 0.01,

                                 max_depth = 10,

                                 subsample = 0.5,

                                 colsample_bytree = 0.5)

    

    regressor_ = regressor.fit(new_train[new_train.date_block_num < 33].drop(['item_cnt_month'], axis=1).values, 

                               new_train[new_train.date_block_num < 33]['item_cnt_month'].values, 

                               eval_metric = 'rmse', 

                               eval_set = [(new_train[new_train.date_block_num < 33].drop(['item_cnt_month'], axis=1).values, 

                                            new_train[new_train.date_block_num < 33]['item_cnt_month'].values), 

                                           (new_train[new_train.date_block_num == 33].drop(['item_cnt_month'], axis=1).values, 

                                            new_train[new_train.date_block_num == 33]['item_cnt_month'].values)

                                          ], 

                               verbose=True,

                               early_stopping_rounds = 20

                              )

    return regressor_
%%time

regressor_ = xgtrain()
predictions = regressor_.predict(new_train[new_train.date_block_num == 34].drop(['item_cnt_month'], axis = 1).values)
submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
submission['item_cnt_month'] = predictions
submission.to_csv('saleslearn.csv', index=False)
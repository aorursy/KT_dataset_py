import numpy as np

from datetime import datetime

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import seaborn as sns

import matplotlib.pyplot as plt

from dateutil.relativedelta import relativedelta

import xgboost as xgb

from xgboost import XGBRegressor

from xgboost import plot_importance

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_predict

from IPython.core.interactiveshell import InteractiveShell

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.preprocessing import StandardScaler
sales_train=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

shops=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

item_categories=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

items=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

test=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
# Viewing first five rows of sales data 

sales_train.head()
unique_shop_ids=sales_train.shop_id.unique()

len(sales_train.shop_id.unique())
# Grouping of Data according to shop_id and date_block_num with total items sold. 

grouped = pd.DataFrame(sales_train.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())
grouped.head()
fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(15,30))

k=0

for i in range(5):

    for j in range(2):

        ids=unique_shop_ids[k:k+6]

        k=k+6

        grouped_d=grouped[grouped['shop_id'].isin(ids)]

        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='shop_id', data=grouped_d,ax=axes[i][j])

plt.title("Item_cnt_day with date_block_num for each shop_id")

plt.show()
# add categories

sales_train = sales_train.set_index('item_id').join(items.set_index('item_id')).drop('item_name', axis=1).reset_index()
# Now we have our *item_category_id* column

sales_train.head()
unique_item_cat=sales_train.item_category_id.unique()

len(unique_item_cat)
group_cat=pd.DataFrame(sales_train.groupby(['item_category_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())
group_cat.head()
fig, axes = plt.subplots(nrows=7, ncols=2, sharex=True, sharey=True, figsize=(15,30))

k=0

for i in range(7):

    for j in range(2):

        ids=unique_item_cat[k:k+6]

        k=k+6

        grouped_cat=group_cat[group_cat['item_category_id'].isin(ids)]

        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='item_category_id', data=grouped_cat,ax=axes[i][j])

plt.title("Item_cnt_day with date_block_num for each item_category_id")

plt.show()
plt.figure(figsize=(10,4))

plt.xlim(sales_train.item_price.min(), sales_train.item_price.max()*1.1)

sns.boxplot(x=sales_train.item_price)
sales_train = sales_train[sales_train.item_price<90000]
plt.figure(figsize=(10,4))

plt.xlim(sales_train.item_cnt_day.min(), sales_train.item_cnt_day.max()*1.1)

sns.boxplot(x=sales_train.item_cnt_day)
sales_train = sales_train[sales_train.item_cnt_day<1100]
sales_train_1 = sales_train.pivot_table(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_day',aggfunc='sum').fillna(0.0)

sales_train_1.head()
sales_train_cleaned = sales_train_1.reset_index()

sales_train_cleaned['shop_id']= sales_train_cleaned.shop_id.astype('str')

sales_train_cleaned['item_id']= sales_train_cleaned.item_id.astype('str')

item_cat = items.merge(item_categories, how="inner", on="item_category_id")

item_cat[['item_id']] = item_cat.item_id.astype('str')

item_cat[['item_category_id']] = item_cat.item_category_id.astype('str')

sales_train_cleaned = sales_train_cleaned.merge(item_cat, how="inner", on="item_id")

sales_train_cleaned = sales_train_cleaned.drop('item_name', axis=1)

sales_train_cleaned = sales_train_cleaned.drop('item_category_name', axis=1)
param = {'max_depth':10, 

         'subsample':1,

         'min_child_weight':0.5,

         'eta':0.3, 

         'num_round':1000, 

         'seed':1,

         'silent':0,

         'eval_metric':'rmse'}
progress = dict()

xgbtrain = xgb.DMatrix(sales_train_cleaned.iloc[:,  (sales_train_cleaned.columns != 33)].values, sales_train_cleaned.iloc[:, sales_train_cleaned.columns == 33].values)

watchlist  = [(xgbtrain,'train-rmse')]

bst = xgb.train(param, xgbtrain)

preds = bst.predict(xgb.DMatrix(sales_train_cleaned.iloc[:,  (sales_train_cleaned.columns != 33)].values))

rmse = np.sqrt(mean_squared_error(preds,sales_train_cleaned.iloc[:, sales_train_cleaned.columns == 33].values))

print(rmse)

apply_df = test

apply_df['shop_id']= apply_df.shop_id.astype('str')

apply_df['item_id']= apply_df.item_id.astype('str')

apply_df = test.merge(sales_train_cleaned, how='left', on = ["shop_id", "item_id"]).fillna(0.0)

cols = apply_df.columns.tolist()

cols = cols[-1:] + cols[:-1]

apply_df=apply_df[cols]

d = dict(zip(apply_df.columns[4:],list(np.array(list(apply_df.columns[4:])) - 1)))

apply_df  = apply_df.rename(d, axis = 1)

preds = bst.predict(xgb.DMatrix(apply_df.iloc[:, (apply_df.columns != 'ID') & (apply_df.columns != -1)].values))

preds = list(map(lambda x: min(20,max(x,0)), list(preds)))

sub_df = pd.DataFrame({'ID':apply_df.ID,'item_cnt_month': preds })
sub_df.to_csv('Submission.csv',index=False)
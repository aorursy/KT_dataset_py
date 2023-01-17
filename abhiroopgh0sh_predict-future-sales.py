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
#---

#**Author**: "Abhiroop Ghosh"

#**Title**: "Future Sales Forecast"

#**Date**: "04/08/2019"

#**output**: "Jupytyr_notebook"

#---

# Kaggle Exercise 

# https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data

# Data set distributed in different files 

# Dataset has the following fields:

#   ID - an Id that represents a (Shop, Item) tuple within the test set

#   shop_id - unique identifier of a shop

#   item_id - unique identifier of a product

#   item_category_id - unique identifier of item category

#   item_cnt_day - number of products sold. You are predicting a monthly amount of this measure

#   item_price - current price of an item

#   date - date in format dd/mm/yyyy

#   date_block_num - a consecutive month number, used for convenience.January 2013 is 0, February 2013 is 1,..., October 2015 is 33

#   item_name - name of item

#   shop_name - name of shop

#   item_category_name - name of item category

##############################################################################

# sales_train.csv -> Data set 

# items.csv -> additional information about the items/products

# item_categories.csv -> additional information about item category 

# shops.csv -> additional information about shops

##############################################################################

# Break the date into its components such as, year, month, day etc.

# Next look at the data, and start asking questions. I give below just a few starting examples.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.metrics import (explained_variance_score, median_absolute_error, mean_absolute_error, mean_squared_error)
train_data = pd.read_csv("../input/sales_train.csv")

items_data = pd.read_csv("../input/items.csv")

item_categories_data = pd.read_csv("../input/item_categories.csv")

test_data = pd.read_csv("../input/test.csv")

sub_sample_data = pd.read_csv("../input/sample_submission.csv")

shops_data = pd.read_csv("../input/shops.csv")
train_data.shape, test_data.shape
train_data.head()
train_data.describe()
train_data.info()
train_data["date"] = pd.to_datetime(train_data["date"], format='%d.%m.%Y')
train_data["date"].head()
train_data["month"] = train_data["date"].dt.month
train_data["month"].head()
train_data["year"] = train_data["date"].dt.year
train_data["year"].head()
train_data = train_data.drop(["date","item_price"], axis=1)
train_data.head()
[count for count in train_data.columns if count not in ["item_cnt_day"]]
train_data["date_block_num"].unique()
train_data.groupby("date_block_num", as_index=False)["item_cnt_day"].sum()
train_data = train_data.groupby([count for count in train.columns if count not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train_data = train_data.rename(columns={'item_cnt_day':'item_cnt_month'})
train_data[['shop_id','item_id','item_cnt_month']].groupby(['shop_id','item_id'], as_index=False)[['item_cnt_month']].mean().head()
train_data = train_data.rename(columns={'item_cnt_day':'item_cnt_month'})
train_data[['shop_id','item_id','item_cnt_month']].groupby(['shop_id','item_id'], as_index=False)[['item_cnt_month']].mean().head()
shop_item_monthly_mean = train_data[['shop_id','item_id','item_cnt_month']].groupby(['shop_id','item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_monthly_mean.head()
shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month':'item_cnt_month_mean'})
train_data.head()
train_data = pd.merge(train_data, shop_item_monthly_mean, how='left', on=['shop_id','item_id'])
train_data.head()
shop_item_prev_month = train_data[train_data['date_block_num']==33][['shop_id','item_id','item_cnt_month']]

shop_item_prev_month.head()
shop_item_prev_month = shop_item_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_month'})
shop_item_prev_month.head()
train_data = pd.merge(train_data, shop_item_prev_month, how='left', on=['shop_id','item_id']).fillna(0.)
shop_item_prev_month.head()
train_data = pd.merge(train_data, shop_item_prev_month, how='left', on=['shop_id','item_id']).fillna(0.)
train_data.head()
train_data = pd.merge(train_data, items_data, how='left', on='item_id')
train_data = pd.merge(train_data, item_categories_data, how='left', on='item_category_id')
test_data["month"] = 11

test_data["year"] = 2015

test_data["date_block_num"] = 34
test_data.head()
test_data = pd.merge(test_data, shop_item_monthly_mean, how='left', on=['shop_id','item_id']).fillna(0.)

test_data = pd.merge(test_data, shop_item_prev_month, how='left', on=['shop_id','item_id']).fillna(0.)
test_data = pd.merge(test_data, items_data, how='left', on='item_id')

test_data = pd.merge(test_data, item_categories_data, how='left', on='item_category_id')
test_data.head()
test_data = pd.merge(test_data, shops, how='left', on='shop_id')

test_data['item_cnt_month'] = 0
plt.subplots(figsize=(8, 6))

sns.heatmap(train_data.corr())
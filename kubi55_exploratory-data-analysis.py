# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

test=pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
train.shape
test.shape
train.head()

test.head()
# Number of NaNs for each object

train.isnull().sum(axis=1).head(20)
# Number of NaNs for each column

train.isnull().sum(axis=0).head()
# `dropna = False` makes nunique treat NaNs as a distinct value

feats_counts = train.nunique(dropna = False)
feats_counts.sort_values()[:10]
nunique = train.nunique(dropna=False)

nunique
shops=pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

items=pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

categories=pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

train.columns

train.shape

shops.shape

#we have 60 shops

items.shape

#we have 22170 item

categories.shape

# we have 84 item categories

#for the training set

prices=train['item_price']

mi=prices.min()

ma=prices.max()

mo=prices.mean()

print(mi,ma,mo)

# Price (min,moy,max) (-1.0, 890.8532326979881, 307980.0, )

max_revenue = (train['item_price'] * train['item_cnt_day']).groupby(train['shop_id']).sum().max()

print(max_revenue)

# max_revenue =235217019.0500024

num_items_constant_price = (train['item_price'].groupby(train['item_id']).nunique() == 1).sum()

print(num_items_constant_price)

#num_items_constant_price= 5926







#sales analysis of their flectuations

import matplotlib.pyplot as plt



t = np.arange(0,34)

nb=train.groupby('date_block_num').sum()

nb_sales=nb['item_cnt_day']

plt.plot(t,nb_sales)

import seaborn as sns

#outliers of price analysis

plt.figure(figsize=(10,4))

plt.xlim(train.item_price.min(), train.item_price.max())

sns.boxplot(x=train.item_price)

#we can see here that there are values that stand alone so we have to omit them so we will take 

# we will take items with price <100000
plt.figure(figsize=(10,4))

plt.xlim(-100, 3000)

sns.boxplot(x=train.item_cnt_day)

#we will take items with cnt_item_day < 1000
train=train[train.item_cnt_day<1000]

train=train[train.item_price<100000]

#abandan negatif price

train=train[train.item_price>0]

train.shape


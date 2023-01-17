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
import matplotlib.pyplot as plt
train_path = "/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv"

test_path = "/kaggle/input/competitive-data-science-predict-future-sales/test.csv"

items_path = "/kaggle/input/competitive-data-science-predict-future-sales/items.csv"

items_categories_path = "/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv"

shops_path = "/kaggle/input/competitive-data-science-predict-future-sales/shops.csv"
train = pd.read_csv(train_path)

test = pd.read_csv(test_path)

items = pd.read_csv(items_path)

items_categories = pd.read_csv(items_categories_path)

shops = pd.read_csv(shops_path)

train.head(3)
test.head(3)
items.head(3)
items_categories.head(3)
train.describe()
train.isna().sum()
train['item_price'][(train['item_price']<=0)].describe()
train['item_cnt_day'][(train['item_cnt_day']<=0)].describe()
test_shop_id = 28

test_item_id = 11330

t = train[(train['item_id']==test_item_id)&(train['shop_id']== test_shop_id)].sort_values(by=['date'])[['date', 'item_price']]

print(t)

t.plot(x = 'date', y = 'item_price', kind='bar')
# test_shop_id = 31

test_item_id = 11330



test_shop_ids = [i for i in range(20, 38)]





fig, axes = plt.subplots(1,8, figsize = (24,4), sharey=True)

i = 0

for test_shop_id in test_shop_ids:

    t = train[(train['item_id']==test_item_id)&(train['shop_id']== test_shop_id)].sort_values(by=['date'])[['date', 'item_price']]

    # t.set_index('date', inplace=True)

    if t.size >0:

        t.plot(x='date', kind='bar', ax=axes[i], title= test_shop_id)

        i+=1

        
train[train['item_price']<1000].groupby(['item_id', 'date_block_num'])['item_price'].mean().unstack(level=0).sample(frac=0.0005, axis=1).plot(kind='line', figsize=(24,8))

train[train['item_price']<1000].groupby(['item_id', 'date_block_num'])['item_price'].mean()
train.groupby(['item_id', 'date_block_num'])['item_cnt_day'].sum().unstack(level=0).fillna(0).sample(frac=0.0005, axis=1).plot(kind='line', figsize=(24,4))
train.groupby('date_block_num')['item_cnt_day'].sum().plot(kind='line', figsize=(24,8))
t_cate = train.join(items[['item_id', 'item_category_id']].set_index('item_id'), on='item_id')

t_cate
t_cate.groupby(['item_category_id', 'date_block_num'])['item_cnt_day'].sum().unstack(level=0).fillna(0).sample(frac=0.05, axis=1).plot(kind='line', figsize=(24,8))
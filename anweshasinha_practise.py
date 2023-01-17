# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns





from datetime import datetime, date

from dateutil.relativedelta import relativedelta



from sklearn.preprocessing import StandardScaler

from math import ceil

from keras.callbacks import LambdaCallback

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense, Activation

from keras.optimizers import RMSprop



%matplotlib.inline
train = pd.read_csv('../input/sales_train.csv')

items = pd.read_csv('../input/items.csv')

categories = pd.read_csv('../input/item_categories.csv')

shops = pd.read_csv('../input/shops.csv')

test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/sample_submission.csv')
train.head(5)
items.head(5)
xyz = pd.merge(train, items)

display(xyz)
categories.head(5)

wxyz = pd.merge(xyz, categories)

display(wxyz)
shops.head(5)

uvwxyz = pd.merge(wxyz, categories)

display(uvwxyz)
test.head(5)

tuvwxyz = pd.merge(uvwxyz, categories)

display(tuvwxyz)
submission.head(5)
tuvwxyz.info()
tuvwxyz.describe()
example_dataframe = pd.DataFrame(tuvwxyz)

print(example_dataframe)
example_dataframe['item_name'].nunique()
tuvwxyz.item_id.nunique()
cde = tuvwxyz.groupby(['item_category_name', 'item_id']) 

print(cde.first())
pd.pivot_table(tuvwxyz, index=['item_category_name'], columns=['item_id'], aggfunc=len, fill_value=0)


data_by_carrier = tuvwxyz.pivot_table(index='date', columns='item_category_name', values='item_id', aggfunc='count')

data_by_carrier.head()

    

test.columns
train.columns
test_shop = test.shop_id.unique() 

train = train[train.shop_id.isin(test_shops)]

train.head(5)
test_items = test.item_id.unique()

train = train[train.item_id.isin(test_items)]

train.head(5)
MAX_BLOCK_NUM = train.date_block_num.max()

MAX_ITEM = len(test_items)

MAX_CAT = len(categories)

MAX_YEAR = 3

MAX_MONTH = 4 # 7 8 9 10

MAX_SHOP = len(test_shops)
#grouped = pd.DataFrame(train.groupby(['shop_id','date_block_num']).sum())

grouped = pd.DataFrame(train.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())

fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))

num_graph = 10

id_per_graph = ceil(grouped.shop_id.max() / num_graph)

print(grouped.head(5)," ",id_per_graph," ", grouped.shop_id.max())
count = 0

data=grouped[np.logical_and(count*id_per_graph <= grouped['shop_id'], grouped['shop_id'] < (count+1)*id_per_graph)]

print(data)
grouped = pd.DataFrame(train.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())

fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))

num_graph = 10

id_per_graph = ceil(grouped.shop_id.max() / num_graph)

count = 0

for i in range(5):

    for j in range(2):

        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='shop_id', data=grouped[np.logical_and(count*id_per_graph <= grouped['shop_id'], grouped['shop_id'] < (count+1)*id_per_graph)], ax=axes[i][j])

        count += 1
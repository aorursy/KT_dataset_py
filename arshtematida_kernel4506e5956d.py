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
pd.options.display.max_rows = 10
df_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

df_test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
df_items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

df_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

df_shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
df_train.head(5)
df_shops.head(5)
df_train['datatype'], df_test['datatype'] = 'training', 'testing'

df_train.insert(0, 'ID', np.nan)

df_test.insert(1, 'date', '01.11.2015')

df_test.insert(2, 'date_block_num', 34)

df_test.insert(df_test.shape[1] - 1, 'item_price', np.nan)

df_test.insert(df_test.shape[1] - 1, 'item_cnt_day', np.nan)

df_data = pd.concat([df_train, df_test], ignore_index=False)
df_data.describe(include='all')
_ = df_data.hist(bins=20, figsize=(10, 6))
df_data.loc[df_data['shop_id'] == 0, 'shop_id'] = 57

df_data.loc[df_data['shop_id'] == 1, 'shop_id'] = 58

df_data.loc[df_data['shop_id'] == 11, 'shop_id'] = 10
df_data.loc[df_data['item_price'] < 0, 'item_price'] = df_data.loc[(df_data['date_block_num'] == 4) & (df_data['shop_id'] == 32) & (df_data['item_price'] > 0), 'item_price'].median()

df_data = df_data[((df_data['item_price'] >= 0) & (df_data['item_price'] <= 100000)) | (df_data['item_price'].isna())]
df_data = df_data[(df_data['item_cnt_day'] <= 1000) | (df_data['item_cnt_day'].isna())]
_ = df_data.hist(bins=20, figsize=(10, 6))
df_data[df_data['item_cnt_day'] == 0].head()
import gc, itertools
list_of_cross = []

for dateblocknum in df_data['date_block_num'].unique():

    shops = df_data.loc[df_data['date_block_num'] == dateblocknum, 'shop_id'].unique()

    items = df_data.loc[df_data['date_block_num'] == dateblocknum, 'item_id'].unique()

    list_of_cross.append(np.array(list(itertools.product(*[[dateblocknum], shops, items]))))

df_cross = pd.DataFrame(np.vstack(list_of_cross), columns=['date_block_num', 'shop_id', 'item_id'])
df_cross.describe(include='all')
df_block = df_data.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False).agg({

    'item_price': 'mean', 'item_cnt_day': sum

}).rename(columns={'item_cnt_day': 'item_cnt_month'})
df_block[df_block['date_block_num'] != 34].describe(include='all')
df_block[df_block['date_block_num'] == 34].describe(include='all')
df_block = pd.merge(df_cross, df_block, how='left', left_on=['date_block_num', 'shop_id', 'item_id'], right_on=['date_block_num', 'shop_id', 'item_id'])
df_block['item_cnt_month'] = df_block['item_cnt_month'].fillna(0)
df_block = pd.merge(df_block, df_items, how='left', left_on='item_id', right_on='item_id')
df_block['item_id'].value_counts()

df_block[df_block['date_block_num'] != 34].describe(include='all')
import matplotlib.pyplot as plt

import seaborn as sns

fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[df_block['item_id'] == 20949].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[df_block['item_id'] == 20949], ax=axes[1])
fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[df_block['item_category_id'] == 71].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[df_block['item_category_id'] == 71], ax=axes[1])

fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[df_block['item_id'] == 8778].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[df_block['item_id'] == 8778], ax=axes[1])
fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[(df_block['item_id'] == 8778) & (df_block['shop_id'] == 31)].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[(df_block['item_id'] == 8778) & (df_block['shop_id'] == 31)], ax=axes[1])
fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[df_block['item_category_id'] == 40].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[df_block['item_category_id'] == 40], ax=axes[1])
df_block.info()
df_block['first_sale_item'] = (df_block['date_block_num'] - df_block.groupby(['item_id'])['date_block_num'].transform('min')).astype(np.int16)

df_block['first_sale_item_shop'] = (df_block['date_block_num'] - df_block.groupby(['item_id', 'shop_id'])['date_block_num'].transform('min')).astype(np.int16)
list_of_first = []

for dateblocknum in df_block['date_block_num'].unique():

    df_first = df_block[(df_block['date_block_num'] < dateblocknum) & (df_block['first_sale_item'] == 0)].groupby(['item_category_id'], as_index=False).agg({'item_cnt_month': 'mean'})

    df_first.insert(0, 'date_block_num', dateblocknum)

    list_of_first.append(df_first)

df_first = pd.concat(list_of_first, ignore_index=True).rename(columns={'item_cnt_month': 'item_cnt_month_by_category_first'})

df_block = pd.merge(df_block, df_first, how='left', on=['date_block_num', 'item_category_id'])
for i in [12, 18, 24, 30]:

    list_of_first = []

    for dateblocknum in df_block['date_block_num'].unique():

        df_first = df_block[(df_block['date_block_num'] >= i) & (df_block['date_block_num'] < dateblocknum) & (df_block['first_sale_item'] == 0)].groupby(['item_category_id'], as_index=False).agg({'item_cnt_month': 'mean'})

        df_first.insert(0, 'date_block_num', dateblocknum)

        list_of_first.append(df_first)

    df_first = pd.concat(list_of_first, ignore_index=True).rename(columns={'item_cnt_month': 'item_cnt_month_by_category_first_month' + str(i) + 'onwards'})

    df_block = pd.merge(df_block, df_first, how='left', on=['date_block_num', 'item_category_id'])
x = df_block[(df_block['first_sale_item'] != 0) & (df_block['date_block_num'] >= 30) & (df_block['date_block_num'] <= 33)].drop(['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'item_category_id'] + [col for col in df_block.columns if col.startswith('item_price')], axis=1)

y = df_block.loc[(df_block['first_sale_item'] != 0) & (df_block['date_block_num'] >= 30) & (df_block['date_block_num'] <= 33), 'item_cnt_month']
from sklearn.preprocessing import LabelEncoder



# import xgboost model class

import xgboost as xgb



# import sklearn model selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split



# import sklearn model evaluation regression metrics

from sklearn.metrics import mean_squared_error
x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=58, test_size=0.25)
del x, y

gc.collect()
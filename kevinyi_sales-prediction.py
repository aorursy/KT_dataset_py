# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/sales_train.csv')

df.head()
df_items = pd.read_csv('../input/items.csv')

df_items.head()
df = pd.merge(df, df_items.drop('item_name', axis = 1), on='item_id')

df.head()
df['amount'] = df['item_price'] * df['item_cnt_day']

df.head()
df = df.drop('date_block_num', axis = 1)

df['month'] = df['date'].apply(lambda x: x.split('.')[1])

df = df.drop('date', axis = 1)

df.head()
df[['shop_id', 'amount']].plot(kind='scatter', x='shop_id', y='amount')
df[['item_price', 'amount']].plot(kind='scatter', x='item_price', y='amount')
df['month'] = df['month'].astype(int)

plt.plot(df[['month', 'amount']].groupby('month').sum())
df[['shop_id','item_id','item_category_id']] = df[['shop_id','item_id','item_category_id']].astype(np.object)

df.info()
df = pd.get_dummies(df, columns=['month','item_category_id'])

df.info()
df.head()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



dtypes = {'store_nbr': np.dtype('int64'),

          'item_nbr': np.dtype('int64'),

          'unit_sales': np.dtype('float64'),

          'onpromotion': np.dtype('O')}



train = pd.read_csv('../input/train.csv', dtype=dtypes)

test = pd.read_csv('../input/test.csv', dtype=dtypes)

stores = pd.read_csv('../input/stores.csv')

items = pd.read_csv('../input/items.csv')

trans = pd.read_csv('../input/transactions.csv')

oil = pd.read_csv('../input/oil.csv')

holidays = pd.read_csv('../input/holidays_events.csv')
df_train = pd.DataFrame(train)

print(df_train.shape)

df_train.head(10)
df_stores = pd.DataFrame(stores)

print(df_stores.shape)

df_stores.head(10)
df_train_merge = df_train.merge(df_stores, on='store_nbr', how='left')

df_train_merge.head(10)
import sys

print(sys.version)
df_items = pd.DataFrame(items)

print(df_items.shape)

df_items.head(10)
df_train_merge = df_train_merge.merge(df_items, on='item_nbr', how='left')

df_train_merge.head(10)
df_items = pd.DataFrame(items)

print(df_items.shape)

df_items.head(10)
df_train_merge = df_train.merge(df_items, on='item_nbr', how='left')

df_train_merge.head(10)
df_items = pd.DataFrame(items)

print(df_items.shape)

df_items.head(10)
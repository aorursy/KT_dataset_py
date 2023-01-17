import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_sell = pd.read_csv('/kaggle/input/retaildataset/sales data-set.csv', header=0)

df_sell.head()
df_store = pd.read_csv('/kaggle/input/retaildataset/stores data-set.csv', header=0)

df_store.head()
df_F = pd.read_csv('/kaggle/input/retaildataset/Features data set.csv', header=0)

df_F.head()
df_sell = pd.merge(df_sell,df_store, on = "Store", how= 'left')

df_sell.head() 
df_sell = df_sell.drop(['Store','Dept','IsHoliday','Type'], axis=1)

df_sell
df_sell['SalesPerSize'] = df_sell['Weekly_Sales'] / df_sell['Size']

df_sell
df_sell = pd.merge(df_sell,df_F, on = "Date", how= 'left')

df_sell
df_sell = df_sell.drop(['Weekly_Sales','Size','Store','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','Unemployment','IsHoliday'], axis=1)

df_sell
df_sell.plot.scatter(x='Temperature',

                      y='SalesPerSize')
import matplotlib.pyplot as plt

plt.scatter('Temperature','SalesPerSize')

plt.show()
#https://www.haya-programming.com/entry/2019/07/18/041646

#こういう感じの図（スキャッター図）にしたい   
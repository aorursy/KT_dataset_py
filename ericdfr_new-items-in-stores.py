import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
# load train

types = {'id': 'int64','item_nbr': 'int32','store_nbr': 'int8','unit_sales': 'float32','onpromotion': bool,}

df_train= pd.read_csv("../input/train.csv", parse_dates = ['date'], dtype = types, infer_datetime_format = True)
# load test

types = {'id': 'int64','item_nbr': 'int32','store_nbr': 'int8','onpromotion': bool,}

df_test = pd.read_csv("../input/test.csv", parse_dates = ['date'], dtype = types, infer_datetime_format = True)
# append dataframes and filter columns

df_t = df_train.append(df_test)

df_t = df_t.filter(items=['date', 'item_nbr', 'store_nbr'])
f, ax = plt.subplots(1, 1, figsize=(20, 10))

df_t_is = df_t.groupby(by=['item_nbr', 'store_nbr']).agg({'date':'min'}).groupby(by='date').size()

df_t_is.plot(ax=ax)
print("median count of new (item, store) by day: {}".format(df_t_is.median()))

df_t_is.sort_values(ascending=False).head(10)
f, ax = plt.subplots(1, 1, figsize=(20, 10))

df_t_i = df_t.groupby(by=['item_nbr']).agg({'date':'min'}).groupby(by='date').size()

df_t_i.plot(ax=ax)
print("median count of new items by day: {}".format(df_t_i.median()))

df_t_i.sort_values(ascending=False).head(10)
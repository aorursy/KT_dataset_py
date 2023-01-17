from datetime import date, timedelta



import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error

import lightgbm as lgb
df_train = pd.read_csv(

    '../input/train.csv', usecols=[1, 2, 3, 4, 5],

    dtype={'onpromotion': bool},

    converters={'unit_sales': lambda u: np.log1p(

        float(u)) if float(u) > 0 else 0},

    parse_dates=["date"],

    skiprows=range(1, 66458909)  # 2016-01-01

)

df_test = pd.read_csv(

    "../input/test.csv", usecols=[0, 1, 2, 3, 4],

    dtype={'onpromotion': bool},

    parse_dates=["date"]  # , date_parser=parser

).set_index(

    ['store_nbr', 'item_nbr', 'date']

)

items = pd.read_csv(

    "../input/items.csv",

).set_index("item_nbr")
items.head()
# Getting the observation from the 11 weeks after 2017-5-31

df_2017 = df_train[df_train.date.isin(

    pd.date_range("2017-05-31", periods=7 * 11))]

df_2017_0 = df_train[df_train.date.isin(

    pd.date_range("2017-05-31", periods=7 * 11))].copy()

del df_train

df_2017.head()
promo_2017_train = df_2017.set_index(

    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(

        level=-1).fillna(False)

promo_2017_train.head()
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)

promo_2017_train.head()
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)

promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)

promo_2017_test.head()
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)

promo_2017_test.head()
# create a big table of on-promotion history, by adding the train to the test

promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)

promo_2017.head()

del promo_2017_test, promo_2017_train
df_2017 = df_2017.set_index(

    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(

        level=-1).fillna(0)

df_2017.columns = df_2017.columns.get_level_values(1)

# get_level_values(1) remove the level 0 index - unit_sales, so that the table could be isomorphic

df_2017.head()
df_2017.head()
df_2017.shape
# items = items.reindex(df_2017.index.get_level_values(1))

items = items.reindex(df_2017.index)

items.head()
items.shape
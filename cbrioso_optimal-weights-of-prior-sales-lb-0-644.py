import numpy as np

import pandas as pd

from sklearn import preprocessing, linear_model, metrics

import gc; gc.enable()

import random

import matplotlib.pyplot as plt

from datetime import timedelta

from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import TheilSenRegressor, BayesianRidge

from sklearn.ensemble import BaggingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_log_error, mean_squared_error

import time

import datetime

import seaborn as sns


# read datasets

dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':str}

data = {

    'tra': pd.read_csv('../input/train.csv', dtype=dtypes, parse_dates=['date']),

    'tes': pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date']),

    'ite': pd.read_csv('../input/items.csv'),

    'sto': pd.read_csv('../input/stores.csv'),

    'trn': pd.read_csv('../input/transactions.csv', parse_dates=['date']),

    'hol': pd.read_csv('../input/holidays_events.csv', dtype={'transferred':str}, parse_dates=['date']),

    'oil': pd.read_csv('../input/oil.csv', parse_dates=['date']),

    }





# Sample down

obj_store = data['tes']['store_nbr'].unique()

obj_item = data['tes']['item_nbr'].unique()

train = data['tra'][(data['tra']['date'].dt.year >= 2017) & (data['tra']['date'].dt.month >= 3) & (data['tra']['store_nbr'].isin(obj_store)) & (data['tra']['item_nbr'].isin(obj_item))]



gc.collect()

train.head()
import warnings

warnings.filterwarnings("ignore")

ini_date = pd.to_datetime('2017-08-16')

nw = 8

ts = data['tes'][['date','item_nbr', 'store_nbr']]

ts['unit_sales'] = np.nan



def back_prop(df1, df2, date):

    dd = str(len(pd.date_range(start=date,end=ini_date, freq='D'))-1)

    df3 =  df2.loc[df2['date'].isin([date + datetime.timedelta(days=x) for x in range(1, 18)]), ['date', 'item_nbr','store_nbr', 'unit_sales']]

    df3['date']= df3['date'] + datetime.timedelta(days=int(dd)-1)  

    df = pd.merge(df1, df3, 

    on = ['item_nbr','store_nbr','date'],

    suffixes = ('', dd),

    how= 'left').fillna(0)

    gc.collect()

    return df



for dt in [ini_date - datetime.timedelta(days=x) for x in range(7*3, 7*nw, 7)]:

    ts = back_prop(ts, train, dt)

gc.collect()

ts.head()
import warnings

warnings.filterwarnings("ignore")

ini_date = pd.to_datetime('2017-07-19')

tr = data['tes'][['date','item_nbr', 'store_nbr']]

dif = len(pd.date_range(start=ini_date, end=tr['date'].min(), freq='D'))

tr['date'] = tr['date'] - datetime.timedelta(days=dif-1) 





def back_prop(df1, df2, date):

    dd = str(len(pd.date_range(start=date,end=ini_date, freq='D'))-1)

    df3 =  df2.loc[df2['date'].isin([date + datetime.timedelta(days=x) for x in range(1, 18)]), ['date', 'item_nbr','store_nbr', 'unit_sales']]

    df3['date']= df3['date'] + datetime.timedelta(days=int(dd)-2)  

    df = pd.merge(df1, df3, 

    on = ['date','item_nbr','store_nbr'],

    suffixes = ('', dd),

    how= 'left').fillna(0)

    gc.collect()

    return df



tr = back_prop(tr, train, ini_date)

for dt in [ini_date - datetime.timedelta(days=x) for x in range(7*3, 7*nw, 7)]:

    tr = back_prop(tr, train, dt)

tr['unit_sales'].mean()

gc.collect()

tr.head()
data_plot = tr[tr['item_nbr'].isin(['105575','105857','112830','116017','116018'])]

data_plot = data_plot[['item_nbr', 'unit_sales', 'unit_sales21', 'unit_sales28', 'unit_sales35', 'unit_sales42']].set_index('item_nbr').stack().reset_index()

data_plot.columns = ['item','lag','sales']

sns.factorplot(x="lag", y="sales", hue= 'item', data = data_plot)

plt.show()
cols = [c for c in tr if c not in ['item_nbr','store_nbr', 'unit_sales', 'date']]

X_train = tr[cols].clip(lower=0)

y_train = tr.unit_sales



regr = linear_model.LinearRegression(fit_intercept = False)

regr.fit(X_train, y_train)

plt.plot(regr.coef_, )

y_pred = regr.predict(ts[cols].clip(lower=0))

y_pred[0:10]

y_pred[8740:8760]

sub = pd.read_csv('../input/sample_submission.csv')

sub['unit_sales'] = y_pred

sub.head(20)
# model submission

sub.to_csv('subma00.csv', index=False)

print('done')
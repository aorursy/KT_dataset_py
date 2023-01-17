import numpy as np

import pandas as pd

import os

import gc

import pickle



import lightgbm as lgb

import xgboost as xgb



from datetime import timedelta



from sklearn import metrics

from sklearn import preprocessing



import matplotlib.pyplot as plt



pd.set_option('display.max_columns', 500)

input_dir = '/kaggle/input/dmc2020/DMC20_Data'
infos = pd.read_csv(f'{input_dir}/infos.csv', sep='|')

orders = pd.read_csv(f'{input_dir}/orders.csv', sep='|')

items = pd.read_csv(f'{input_dir}/items.csv', sep='|')
N_WEEKS = 2



# Creating date feature in format YYYY-MM-DD

orders['time'] = pd.to_datetime(orders['time'])

orders['date'] = orders['time'].dt.date



# Making sure to get Kaggle train data only

orders = orders[orders['date'] < pd.to_datetime('2018-06-16')].copy()

print(orders['date'].max())



# Tranforming date to nº of days since first date

basedate = orders['date'].min()

orders['date'] = (orders['date'] - basedate).dt.days



# Transforming date to 'nº of n-week blocks - we'll work with 2 weeks for now

orders['date'] = orders['date'] // (7 * N_WEEKS)
# Grouping orders by day and itemID, getting the sum of orders and mean of salesPrice

orders_by_date = orders.groupby(['date', 'itemID'], as_index=False).agg({'order':'sum', 

                                                                            'salesPrice':'mean'})

# Creating dataframe in the usual timeseries format just to take a look

timeseries = orders_by_date.pivot(index='itemID', columns='date')['order']

timeseries = timeseries.fillna(0)
timeseries
sub = orders_by_date[orders_by_date['date'] >= 8].copy()

sub = sub.groupby('itemID', as_index=False)['order'].mean()



sub = items[['itemID']].merge(sub, on='itemID', how='left')

sub = sub.fillna(0)



sub.columns = ['itemID', 'demandPrediction']



sub['demandPrediction'] = sub['demandPrediction'].round(0).astype(int)

sub.to_csv('submission.csv', index=False)
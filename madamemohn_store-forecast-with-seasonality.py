import ipywidgets as widgets

from ipywidgets import interact

from itertools import product

import matplotlib.pyplot as plt

import numpy as np

import os

import qgrid

from pathlib import Path

import pandas as pd

import sklearn

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from tensorflow import keras

import tensorflow.keras.backend as K

import tensorflow as tf
import cufflinks as cf

import plotly.offline as py

import plotly

import plotly.graph_objs as go
qgrid.enable()
CURRENT_PATH = Path(os.getcwd())

OUTPUT_PATH = CURRENT_PATH / 'output'

os.makedirs(OUTPUT_PATH, exist_ok=True)
!ls
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train_raw = pd.read_csv(

    '/kaggle/input/demand-forecasting-kernels-only/train.csv', parse_dates=['date'])

df_test_raw = pd.read_csv(

    '/kaggle/input/demand-forecasting-kernels-only/test.csv', parse_dates=['date'])
df_train = df_train_raw.copy(deep=True)

df_test = df_test_raw.copy(deep=True)
df_train = df_train.set_index(['date', 'store', 'item']).sort_index()

df_test = df_test.set_index(['date', 'store', 'item']).sort_index()



grid = qgrid.show_grid(df_train)

grid
# initialize prediction DF

prediction = pd.DataFrame(0, index=df_train.index, columns=['prediction'])  
cf.go_offline() # required to use plotly offline (no account required).

py.init_notebook_mode() # graphs charts inline (IPython).
df = df_train.copy(deep=True)  # to make it a function later

dates = df.index.get_level_values(0)

df_seasonality_train = pd.DataFrame(index=df.index)

df_seasonality_train['month'] = df.groupby(dates.month).transform('mean')

df_seasonality_train['year'] = df.groupby(dates.year).transform('mean')

df_seasonality_train['week'] = df.groupby(dates.week).transform('mean')

df_seasonality_train['dayofweek'] = df.groupby(dates.dayofweek).transform('mean')

t0 = df.index.get_level_values('date')[0]

df_seasonality_train['day'] = (df.index.get_level_values('date') - t0).days
store_index, item_index = df_train.index.names.index('store'), df_train.index.names.index('item')

stores, items = df_train.index.levels[store_index], df_train.index.levels[item_index]
df_seasonality_train
dates_train = df_train.index.get_level_values('date')

yearly_trend = df_train.groupby(dates_train.year).mean()

yearly_trend.loc[2018, 'sales'] = 61  # add 2018 by hand
yearly_trend.plot()
df_test
def get_seasonality_df(df_pred):

    df_seasonality = pd.DataFrame(index=df_pred.index)

    dates_predict = df_pred.index.get_level_values('date')

    

    for grouper in ['month', 'year', 'week', 'dayofweek', 'day']:

        if grouper == 'year':

            mean_values = yearly_trend

        else:

            mean_values = df_train.groupby(getattr(dates_train, grouper)).mean()

        df_seasonality[grouper] = mean_values.reindex(

            getattr(dates_predict, grouper)).values

    df_seasonality['day'] = (df_pred.index.get_level_values('date') - t0).days

    return df_seasonality



df_seasonality_train = get_seasonality_df(df_train)

df_seasonality_test = get_seasonality_df(df_test)
def select_store_item(df, store, item):

    return df.loc[(slice(None), store, item), :]



def fit_seasonality_by_item_and_store(df, df_seasonality=None):

    regressions = dict()

    

    if df_seasonality is None:

        df_seasonality = get_seasonality_df(df)

        

    for store, item in product(stores, items):

        reg = LinearRegression()

        df_seasonality_selected = select_store_item(df_seasonality, store, item)

        reg.fit(X=df_seasonality_selected, y=select_store_item(df, store, item))

        regressions[store, item] = reg



    return regressions



regressions = fit_seasonality_by_item_and_store(df_train, df_seasonality_train)
def predict_by_item_and_store(df, regressions, df_seasonality=None):

    if df_seasonality is None:

        df_seasonality = get_seasonality_df(df)



    prediction = pd.DataFrame(columns=['prediction'], index=df.index)

    

    for store, item in product(stores, items):

        df_seasonality_selected = select_store_item(df_seasonality, store, item)    

        reg = regressions[store, item]

        prediction.loc[(slice(None), store, item), :] = reg.predict(df_seasonality_selected)

    

    prediction = round(prediction).astype(int)

    return prediction

    

    

prediction_test = predict_by_item_and_store(df_test, regressions, df_seasonality_test)

prediction_train = predict_by_item_and_store(df_train, regressions, df_seasonality_train)
prediction_train.to_csv(OUTPUT_PATH / 'linear_regression_by_store_and_product_train.csv')

prediction_test.to_csv(OUTPUT_PATH / 'linear_regression_by_store_and_product_test.csv')
prediction = prediction_train

df = pd.concat([df_train, prediction], axis=1)



def plot_sales(store, item, resample, groupby, cols):

    if cols:

        df_selected = pd.DataFrame(index=df.index)

        df_selected = df.loc[:, cols]  # select columns



        df_selected = df_selected.loc[(slice(None), store, item), :]  # select store and item

        df_selected = df_selected.set_index(df_selected.index.droplevel([1, 2]))  # drop store and item level in multiindex

        df_selected = df_selected.resample(resample).mean()



        if groupby == 'day':

            days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']

            df_selected = df_selected.groupby(df_selected.index.day_name()).mean().reindex(days)

        elif groupby == 'month':

            df_selected = df_selected.groupby(df_selected.index.month_name()).mean()

        elif groupby == 'week':

            df_selected = df_selected.groupby(df_selected.index.week).mean()



        df_selected.iplot()

    

interact(plot_sales, store=(1,10), item=(1,50), 

         resample=['d', 'W', 'Y'], 

         groupby=['None', 'day', 'week', 'month'],

         cols=widgets.SelectMultiple(options=df.columns),

         continuous_update=False

        );
def smape(series1, series2):

    n = len(series1)

    assert n == len(series2)

    denominator = (np.abs(series1) + np.abs(series2))/2

    denominator.where(denominator != 0, 1, inplace=True)

    return 100 / n * np.sum(np.abs(series1 - series2) / denominator)
smape(df_train['sales'], prediction_train['prediction'])
prediction_test
df_test
df_test.index.equals(prediction_test.index)
out = pd.DataFrame(data=prediction_test['prediction'].values, 

                   index=df_test['id'].astype(int), 

                   columns=['sales'])
out.sort_index(inplace=True)
out.to_csv('submission.csv')
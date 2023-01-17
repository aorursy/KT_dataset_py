import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# import files

sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

items_info = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

icategories_info = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

shops_info = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')



print(sales_train.head())
# add categories

sales_train = sales_train.set_index('item_id').join(items_info.set_index('item_id')).drop('item_name', axis=1).reset_index()

print(sales_train.head())
# select the columns we're interested in

new_sales_train = sales_train[['date_block_num', 'shop_id', 'item_id','item_category_id', 'item_price', 'item_cnt_day']]

# print(new_sales_train.head())



# aggregate by month

monthly_sales = sales_train.groupby(['date_block_num', 'shop_id', 'item_id', 'item_category_id'])['item_price', 'item_cnt_day'].agg({

    'item_price': 'mean',

    'item_cnt_day': 'sum'

}).reset_index()

print(monthly_sales.head())
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs
ts = sales_train.groupby(['date_block_num'])['item_cnt_day'].sum()

plt.plot(ts)

plt.xlabel('Time')

plt.ylabel('Sales')

# plt.show()
import statsmodels.api as sm

res = sm.tsa.seasonal_decompose(ts.values, freq=12, model='multiplicative')

fig = res.plot()

fig.show()
# augmented Dicky-Fuller Test for stationarity

dftest = adfuller(ts, autolag='AIC')

print(dftest)
# remove trend and seasonalization

from pandas import Series as Series



# create a differenced series

def difference(dataset, interval=1):

    diff = list()

    for i in range(interval, len(dataset)):

        value = dataset[i] - dataset[i - interval]

        diff.append(value)

    return Series(diff)



# invert differenced forecast

def inverse_difference(last_ob, value):

    return value + last_ob



new_ts = difference(ts)

plt.plot(new_ts)
dftest = adfuller(new_ts) # p-value <= 0.5 means data is stationary

print(dftest)
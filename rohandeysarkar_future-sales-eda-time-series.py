import os

os.listdir('../input/competitive-data-science-predict-future-sales')
import numpy as np 

import pandas as pd

import random as rd 

import math

import datetime 

import matplotlib.pyplot as plt 

import seaborn as sns 



from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs



import warnings

warnings.filterwarnings("ignore")
shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

sales = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

item_cat = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")



sub = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")



test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
items.head()
topItemCats = items["item_category_id"].value_counts().index[:20]

topItemCatsValues = items["item_category_id"].value_counts().values[:20]



plt.figure(figsize=(18, 6))

ax = sns.barplot(topItemCats, topItemCatsValues, alpha=0.8)

plt.title("Top 20 item catgories sold")

plt.ylabel('Items values', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
sales.info()
sales["date"] = pd.to_datetime(sales["date"])



sales.head()
price_idx = sales["item_price"].value_counts().index[:10]

unitsSold = sales["item_price"].value_counts().values[:10]



plt.figure(figsize=(18, 6))

ax = sns.barplot(price_idx, unitsSold, order=price_idx, alpha=0.8)

plt.title("Top 10 most sold amount")

plt.ylabel('Units Sold', fontsize=12)

plt.xlabel('Price', fontsize=12)

plt.show()
shop_id_idx = sales["shop_id"].value_counts().index[:10]

unitsSold = sales["shop_id"].value_counts().values[:10]



plt.figure(figsize=(18, 6))

ax = sns.barplot(shop_id_idx, unitsSold, order=shop_id_idx, alpha=0.8)

plt.title("Top 10 shops with most sales")

plt.ylabel('Units Sold', fontsize=12)

plt.xlabel('Shop ID', fontsize=12)

plt.show()
sales["year"] = sales["date"].dt.year

sales["month"] = sales["date"].dt.month

sales["day"] = sales["date"].dt.day
plt.figure(figsize=(8,4))

sns.countplot(y="year", data=sales)

plt.title("Total sales yearly")
# We want to predict total products sold, so we are grouping no. of products sold on each "date_block_num"

ts = sales.groupby(["date_block_num"])["item_cnt_day"].sum()

ts
plt.figure(figsize=(16, 6))

plt.title('Total sales of the company')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(ts)
plt.figure(figsize=(16, 6))

plt.plot(ts.rolling(12).mean().values, label='Rolling mean')

plt.plot(ts.rolling(12).std().values, label='Rolling std')

plt.legend()
from statsmodels.tsa.seasonal import seasonal_decompose



res = sm.tsa.seasonal_decompose(ts.values, freq=12, model="multiplicative")

fig = res.plot()
res = sm.tsa.seasonal_decompose(ts.values, freq=12, model="additive")

fig = res.plot()
def test_stationarity(data):

    dftest = adfuller(data, autolag="AIC")

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of observation Used'])

    for key, val in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = val

    print(dfoutput)

    

test_stationarity(ts)
log_transform = ts

log_transform = log_transform.apply(lambda x: math.log(1 + x))

log_transform = pd.DataFrame(log_transform)

diff = log_transform - log_transform.shift(1)

diff = diff.fillna(0) 

test_stationarity(diff)
plt.figure(figsize=(16,16))



plt.subplot(311)

plt.title('Original')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(ts)



plt.subplot(312)

plt.title('After De-trend')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(diff)

plt.show()
#ACF and PACF plots:

from statsmodels.tsa.stattools import acf, pacf



lag_acf = acf(diff, nlags=20)

lag_pacf = pacf(diff, nlags=20, method='ols')



plt.figure(figsize=(16, 7))

#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf, marker="o")

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(diff)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(diff)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')





#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf, marker="o")

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(diff)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(diff)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
from statsmodels.tsa.arima_model import ARIMA



model = ARIMA(ts, order=(1, 1, 1))

results_AR = model.fit(disp=-1)
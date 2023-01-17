# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

sales = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
!ls /kaggle/input/*

# Basic packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rd # generating random numbers

import datetime # manipulating date formats

# Viz

import matplotlib.pyplot as plt # basic plotting

import seaborn as sns # for prettier plots





# TIME SERIES

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs





# settings

import warnings

warnings.filterwarnings("ignore")

sales.head()
shops.head()
# The datatype of date field is object. It needs to be chaged to Date-Time format. 

sales.info()
#formatting the date column correctly

sales.date = sales.date.apply(lambda x : datetime.datetime.strptime(x,'%d.%m.%Y'))

print(sales.info())
monthly_sales = sales.groupby(['date_block_num','shop_id','item_id'])['date','item_price','item_cnt_day'].agg({'date': ['min','max'],'item_price':'mean','item_cnt_day':'sum'})
monthly_sales.head(100)
x = items.groupby('item_category_id').agg({'item_id':'count'})

#x=x.sort_values(by='item_id',ascending=False)

x=x.reset_index()

print(x)





plt.figure(figsize=(25,15))

ax = sns.barplot(x.item_category_id, x.item_id)

plt.title("No. Items per Category")

plt.ylabel('# of items', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
ts = sales.groupby('date_block_num')['item_cnt_day'].sum()

ts_sns = ts.reset_index()

plt.figure(figsize=(15,10))

ax = sns.lineplot(ts_sns.date_block_num,ts_sns.item_cnt_day)

plt.show()
plt.figure(figsize=(16,6))

plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');

plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling sd');

plt.legend();
import statsmodels.api as sm

# multiplicative

res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="multiplicative")

fig = res.plot()

fig.show()
# Additive model

res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")

#plt.figure(figsize=(16,12))

fig = res.plot()

#fig.show()
# Stationarity tests

def test_stationarity(timeseries):

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)

    

test_stationarity(ts)



# to remove trend

from pandas import Series as Series

# create a differenced series

def difference(dataset, interval=1):

    diff = list()

    for i in range(interval, len(dataset)): #12 to 32

        value = dataset[i] - dataset[i - interval]

        diff.append(value)

    return Series(diff)



# invert differenced forecast

def inverse_difference(last_ob, value):

    return value + last_ob

ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.astype('float')

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

new_ts_a=difference(ts)

plt.plot(new_ts_a)

plt.plot()



plt.subplot(313)

plt.title('After De-seasonalization')

plt.xlabel('Time')

plt.ylabel('Sales')

new_ts=difference(ts,12)       # assuming the seasonality is 12 months long

plt.plot(new_ts)

plt.plot()
# now testing the stationarity again after de-seasonality

test_stationarity(new_ts)
test_stationarity(new_ts_a)
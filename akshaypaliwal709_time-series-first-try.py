# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime # manipulating date formats

import random as rd # generating random numbers

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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

!ls ../input/*

# Any results you write to the current directory are saved as output.
#importing the data

sales=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")



# settings

import warnings

warnings.filterwarnings("ignore")



item_cat=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

item=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

sub=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")

test=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
sales.head()
sales.date = sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

print(sales.info())
monthly_sales = sales.groupby(["date_block_num", "shop_id", "item_id"])["date", "item_price", "item_cnt_day"].agg({"date":["min", "max"], "item_price":"mean", "item_cnt_day":"sum"})

monthly_sales.head(20)
x = item.groupby(["item_category_id"]).count()

x = x.sort_values(by = "item_id", ascending = False)

x = x.iloc[0:10].reset_index()

x

plt.figure(figsize = (8,4))

ax = sns.barplot(x.item_category_id, x.item_id, alpha = 0.8)

plt.title("Items/Category")

plt.ylabel("# of items", fontsize = 12)

plt.xlabel("Category", fontsize=12)

plt.show()
ts = sales.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.astype('float')

plt.figure(figsize=(16,8))

plt.title("Total sales of the company")

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(ts)
plt.figure(figsize=(16,6))

plt.plot(ts.rolling(window=12, center = False).mean(), label = "Rolling Mean");

plt.plot(ts.rolling(window=12, center = False).std(), label = "Rolling sd")

plt.legend();
import statsmodels.api as sm

res = sm.tsa.seasonal_decompose(ts.values, freq = 12, model = "multiplicative")

fig = res.plot()
import statsmodels.api as sm

res = sm.tsa.seasonal_decompose(ts.values, freq = 12, model = "multiplicative")

fig = res.plot()
def test_stationarity(timeseries):

    print("Results of Dickey - Fuller Test :")

    dftest = adfuller(timeseries, autolag = 'AIC')

    dfoutput = pd.Series(dftest[0:4], index = ["test statistic", 'p-value', 'Lags used', 'Number of observations used'])

    for key, value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)



test_stationarity(ts)
#no co relation 

# now remove trend 

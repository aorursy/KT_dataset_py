!ls ../input/*
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

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs





# settings

import warnings

warnings.filterwarnings("ignore")



# Read all data 

sales=pd.read_csv("../input/sales_train.csv")





item_cat=pd.read_csv("../input/item_categories.csv")

item=pd.read_csv("../input/items.csv")

sub=pd.read_csv("../input/sample_submission.csv")

shops=pd.read_csv("../input/shops.csv")

test=pd.read_csv("../input/test.csv")
sales.head(2)
#formatting the date column correctly

sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

# check

print(sales.info())
# Aggregate to monthly level the required metrics



monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"])[

    "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})

monthly_sales.head(20)
# number of items per cat 

x=item.groupby(['item_category_id']).count()

x=x.sort_values(by='item_id',ascending=False)

x=x.iloc[0:10].reset_index()

x

# #plot

plt.figure(figsize=(8,4))

ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)

plt.title("Items per Category")

plt.ylabel('# of items', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
#date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33

ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.astype('float')

plt.figure(figsize=(16,8))

plt.title('Total Sales of the company')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(ts);
plt.figure(figsize=(16,6))

plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');

plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling sd');

plt.legend();
import statsmodels.api as sm

# multiplicative

res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="multiplicative")

#plt.figure(figsize=(16,12))

fig = res.plot()

#fig.show()
from pylab import rcParams

rcParams['figure.figsize'] = 16, 8



res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")

fig = res.plot()

plt.show()
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

    for i in range(interval, len(dataset)):

        value = dataset[i] - dataset[i - interval]

        diff.append(value)

    return Series(diff)



# invert differenced forecast

def inverse_difference(last_ob, value):

    return value + last_ob



new_ts = difference(ts)

test_stationarity(new_ts)
fig = plt.figure(figsize=(16,10))

layout = (2,2)

ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)

acf_ax = plt.subplot2grid(layout, (1, 0))

pacf_ax = plt.subplot2grid(layout, (1, 1))

new_ts.plot(ax=ts_ax)

ts_ax.set_title("1st order differencing")

plot_acf(ts, ax=acf_ax, alpha=0.5)

plot_pacf(ts, ax=pacf_ax, alpha=0.5)

plt.tight_layout()

model = ARIMA(ts.values, order=(1,2,3))

model_fit = model.fit(disp=0)

print(model_fit.summary())
import itertools

p = q = d = range(0, 4)

pdq = itertools.product(p, d, q)

for param in pdq:

    try:

        mod = ARIMA(ts.values,order=param)

        results = mod.fit()

        print('ARIMA{} - AIC:{}'.format(param, results.aic))

    except:

        continue
model = ARIMA(ts.values, order=(1,2,2))

result = model.fit()

print(result.summary())
pred = result.predict(start = 24, end=40)

prediction = pd.Series(pred, index=range(24,41,1))

#prediction
new_ts.plot()

prediction.plot()
# Plot residual errors

residuals = pd.DataFrame(result.resid)

fig, ax = plt.subplots(1,2)

residuals.plot(title="Residuals", ax=ax[0])

residuals.plot(kind='kde', title='Density', ax=ax[1])

plt.show()
min_aic = 1000

import itertools

p = q = d = range(0, 3)

pdq = itertools.product(p, d, q)

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(ts.values,

                                            order=param,

                                            seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

            if results.aic < min_aic:

                min_aic = results.aic

        except:

            continue

min_aic
#ARIMA(2, 2, 0)x(1, 1, 0, 12)

mod = sm.tsa.statespace.SARIMAX(ts.values,

                                            order=(2,2,0),

                                            seasonal_order=(1,1,0,12),

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)

result = mod.fit()

result.summary()
residuals = pd.DataFrame(result.resid)

fig, ax = plt.subplots(1,2)

residuals.plot(title="Residuals", ax=ax[0])

residuals.plot(kind='kde', title='Density', ax=ax[1])

plt.show()
pred = result.predict(start = 24, end=40)

prediction = pd.Series(pred, index=range(24,41,1))

ts.plot()

prediction.plot()
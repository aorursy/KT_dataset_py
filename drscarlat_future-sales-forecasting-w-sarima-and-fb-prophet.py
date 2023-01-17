import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rd # generating random numbers

import datetime # manipulating date formats

from sklearn.metrics import mean_squared_error

from numpy import sqrt



import matplotlib.pyplot as plt # basic plotting

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 10, 6

import seaborn as sns # for prettier plots



from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.ar_model import AR

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX



from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scsor 



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Load data



sales=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

item_cat=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

item=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

shops=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")



print('sales ' , sales.shape)

print('item_cat ' , item_cat.shape)

print('item ' , item.shape)

print('shops ' , shops.shape)

# formatting the date column from object to date time



print(sales.info())

sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

print(sales.info())
# Group by total monthly sales ...34 months



ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.astype('float')

plt.figure(figsize=(10,6))

plt.title('Total sales of company')

plt.xlabel('Months')

plt.ylabel('Sales')

indexedDataset = pd.DataFrame(ts)

#indexedDataset.head()

plt.plot(indexedDataset)

plt.show()
MyWindow = 3



#Determine rolling statistics

rolmean = indexedDataset.rolling(window=MyWindow).mean() #window size 12 denotes 12 months, giving rolling mean at yearly level

rolstd = indexedDataset.rolling(window=MyWindow).std()



#Plot rolling statistics

orig = plt.plot(indexedDataset, color='blue', label='Original')

mean = plt.plot(rolmean, color='red', label='Rolling Mean')

std = plt.plot(rolstd, color='black', label='Rolling Std')

plt.legend(loc='best')

plt.title('Rolling Mean & Standard Deviation')

plt.show(block=False)
# decompose into trend, seasonality and residuals

res = sm.tsa.seasonal_decompose(indexedDataset.values,freq=MyWindow,model="additive")

#plt.figure(figsize=(16,12))

fig = res.plot()
#Perform Augmented Dickey–Fuller test for stationarity



print('Results of Dickey Fuller Test:')

dftest = adfuller(indexedDataset['item_cnt_day'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

    

print(dfoutput)
dataAR = list(indexedDataset.item_cnt_day.values)

len(dataAR)
# AR model 



model = AR(dataAR)

model_fit = model.fit()

# make prediction

yhat = model_fit.predict(12, len(dataAR)+ 18) # predict N ahead of the last one



dataList = list(dataAR)

yhatList = list(yhat)



plt.style.use('seaborn-poster')

plt.figure()

plt.plot(dataList, label='Original')

plt.plot(yhatList, ls='--', label='Predicted')

plt.legend(loc='best')

plt.title('AR model')

plt.show()
rmse = sqrt(mean_squared_error(dataList,yhatList[0:34]))

print('AR RMSE: %.1f' % rmse)
# Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots

# Get p and q for ARIMA



plt.figure(figsize=(15,10))

ax = plt.subplot(211)

sm.graphics.tsa.plot_acf(dataAR, lags=12, ax=ax)

ax = plt.subplot(212)

sm.graphics.tsa.plot_pacf(dataAR, lags=12, ax=ax)

#plt.tight_layout()

plt.show()
# ARIMA model



model = ARIMA(dataAR, order=(2, 1, 1))

model_fit = model.fit(disp=False)

# make prediction

yhat = model_fit.predict(1, len(dataAR)+6, typ='levels')



dataList = list(dataAR)

yhatList = list(yhat)



plt.style.use('seaborn-poster')

plt.figure()

plt.plot(dataList, label='Original')

plt.plot(yhatList, ls='--', label='Predicted')

plt.legend(loc='best')

plt.title('ARIMA model')

plt.show()
rmse = sqrt(mean_squared_error(dataList,yhatList[0:34]))

print('ARIMA RMSE: %.1f' % rmse)
# SARIMA



model = SARIMAX(dataAR, order=(2, 1, 1), seasonal_order=(2,1,1,3))

model_fit = model.fit(disp=False)

# make prediction

yhat = model_fit.predict(1, len(dataAR)+6)



dataList = list(dataAR)

yhatList = list(yhat)



plt.style.use('seaborn-poster')

plt.figure()

plt.plot(dataList, label='Original')

plt.plot(yhatList, ls='--', label='Predicted')

plt.legend(loc='best')

plt.title('SARIMAX model')

plt.show()
rmse = sqrt(mean_squared_error(dataList,yhatList[0:34]))

print('SARIMA RMSE: %.1f' % rmse)
# adding the dates to the Time-series as index ... required by FB Prophet

ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

ts=ts.reset_index()

ts.head()
from fbprophet import Prophet

# prophet REQUIRES a pandas df at the below config 

# date column named as DS and the value column as Y

ts.columns=['ds','y']

model = Prophet(yearly_seasonality=True, weekly_seasonality=True) # instantiate Prophet with only yearly seasonality 

model.fit(ts) # fit the model with the ts dataframe
# predict for six months in the furure and MS - monthly = frequency

future = model.make_future_dataframe(periods = 6, freq = 'MS')  

forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
dataList = list(indexedDataset.item_cnt_day.values)
plt.style.use('seaborn-poster')

plt.figure()

plt.plot(dataList, label='Original')

plt.plot(forecast['yhat'], ls='--', label="Predicted")

plt.legend(loc='best')

plt.title('FB Prophet model')

plt.show()

# RMSE



rmse = sqrt(mean_squared_error(dataList,forecast['yhat'][0:34]))

print('Val RMSE: %.1f' % rmse)
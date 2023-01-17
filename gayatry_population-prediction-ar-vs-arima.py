"""Importing required libraries"""

import numpy as np

import pandas as pd

from pandas import Series,DataFrame



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
plt.rcParams['figure.figsize'] = (10,10)

plt.style.use('ggplot')
data = pd.read_csv('/kaggle/input/population-time-series-data/POP.csv')

data
data = data.drop(['realtime_start','realtime_end'],axis=1)
"""Converting the date column into datetime object and setting it as index"""

data['date'] = pd.to_datetime(data['date'])

data.set_index('date',inplace=True)

data.head()
data.describe()
data.plot()
pd.plotting.autocorrelation_plot(data['value'])
data['value'].corr(data['value'].shift(300))
from statsmodels.tsa.seasonal import seasonal_decompose
decomposed = seasonal_decompose(data['value'])

x = decomposed.plot()
from statsmodels.tsa.stattools import adfuller
print('Significance level : 0.05')

adf = adfuller(data['value'])

print(f'ADF test static is {adf[1]}')
data['stationary'] = data['value'].diff()
data['stationary'].plot()
print('Significance level : 0.05')

adf = adfuller(data['stationary'].dropna())

print(f'ADF test static is {adf[1]}')
data['stationary2'] = data['stationary'].diff()
data['stationary2'].plot()
print('Significance level : 0.05')

adf = adfuller(data['stationary2'].dropna())

print(f'ADF test static is {adf[1]}')
t = seasonal_decompose(data['stationary2'].dropna())

x = t.plot()
from statsmodels.tsa.ar_model import AR
"""Creating train & Test dataset"""



X = data['stationary2'].dropna()



train_df,test_df = X[1:(len(X)-280)],X[(len(X)-280):]
"""Training the model"""



model = AR(train_df)

model_fitted = model.fit()
print(f'The chosen lag value is {model_fitted.k_ar}',end='\n \n')



print(f'The coefficents are {model_fitted.params}')
"""Make predictions"""



predictions = model_fitted.predict(start=len(train_df),

                                   end=len(train_df)+len(test_df)-1,

                                   dynamic=False)
"""Comparing with actual data"""



compare_df = pd.concat([test_df,predictions],axis=1).rename(columns={'stationary2': 'actual', 0:'predicted'})
compare_df.plot()
from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf
data = pd.read_csv('/kaggle/input/population-time-series-data/POP.csv')

data = data.drop(['realtime_start','realtime_end'],axis=1)

data
fig,axes = plt.subplots(3,2)



x = axes[0,0].plot(data['value']); axes[0,0].set_title('Original Series')

a = plot_acf(data['value'].values,ax=axes[0,1])



y = axes[1,0].plot(data['value'].diff()); axes[1,0].set_title('1st Difference')

b = plot_acf(data['value'].diff().dropna(),ax=axes[1,1])



z = axes[2,0].plot(data['value'].diff().diff()); axes[2,0].set_title('2nd Difference')

c = plot_acf(data['value'].diff().diff().dropna(),ax=axes[2,1])
plt.rcParams.update({'figure.figsize':(9,3),'figure.dpi':120})



fig,axes = plt.subplots(1,2)



a = axes[0].plot(data['value'].diff()); axes[0].set_title('1st Difference')

b = plot_pacf(data['value'].diff().dropna(),ax=axes[1])



plt.show()
fig,axes = plt.subplots(1,2)



a = axes[0].plot(data['value'].diff()); axes[0].set_title('1st Difference')

b = plot_acf(data['value'].diff().dropna(),ax=axes[1])
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(data['value'].diff().dropna(),(1,1,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)



fig,axes = plt.subplots(1,2)



residuals.plot(title='Residuals',ax= axes[0])

residuals.plot(kind= 'kde', title='Density',ax= axes[1])
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':100})



x = model_fit.plot_predict(dynamic=False)

plt.show()
#Train & Test Data

train_data = data['value'][:500]

test_data = data['value'][500:]
model1 = ARIMA(train_data,order=(1,1,2))

model_fitted1 = model1.fit(disp= -1)
fc,se,conf = model_fitted1.forecast(316)
fc_series = Series(fc,index=test_data.index)

lower_series = Series(conf[:,0],index=test_data.index)

upper_series = Series(conf[:,1],index=test_data.index)
plt.figure(figsize=(12,5), dpi=100)



plt.plot(train_data,label='Training')

plt.plot(test_data,label='Actual')

plt.plot(fc_series,label='Forcast',color='green')



plt.fill_between(lower_series.index,lower_series,upper_series,color='k',alpha=.15)



plt.title('Actual Vs Forcast')

plt.legend(loc='upper left')
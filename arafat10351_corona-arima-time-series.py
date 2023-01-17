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
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df = df.sort_values(by='ObservationDate', ascending=False)
#Province and State Wise Time Series Analysis





import seaborn as sns



dfc = df[['ObservationDate', 'Confirmed']]



dfc['Date'] = dfc['ObservationDate']

dfc['Date'] = dfc['Date'].astype('datetime64')

cases = dfc.set_index('Date')

cases.index
y = cases['Confirmed'].resample('D').mean()

y
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import matplotlib

plt.style.use('ggplot')

import warnings

import itertools



y.plot(figsize=(15, 6))

plt.show()
from pylab import rcParams

import statsmodels.api as sm

rcParams['figure.figsize'] = 16, 10

decomposition = sm.tsa.seasonal_decompose(y, model='multiplicative')

fig = decomposition.plot()

plt.show()
from statsmodels.tsa.stattools import adfuller

result = adfuller(y)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

type(y)

d = pd.DataFrame(y)

d = d.reset_index()
# import numpy as np, pandas as pd

# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# import matplotlib.pyplot as plt

# plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})



# Original Series

fig, axes = plt.subplots(3, 2, sharex=True)

axes[0, 0].plot(d.Confirmed); axes[0, 0].set_title('Original Series')

plot_acf(y, ax=axes[0, 1])



# 1st Differencing

axes[1, 0].plot(d.Confirmed.diff()); axes[1, 0].set_title('1st Order Differencing')

plot_acf(d.Confirmed.diff().dropna(), ax=axes[1, 1])



# 2nd Differencing

axes[2, 0].plot(d.Confirmed.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')

plot_acf(d.Confirmed.diff().diff().dropna(), ax=axes[2, 1])



plt.show()
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(y)
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(y, order=(9,2,1))

model_fit = model.fit(disp=0)

print(model_fit.summary())
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(y, order=(5,2,1))

model_fit = model.fit(disp=0)

print(model_fit.summary())
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(y, order=(4,2,1))

model_fit = model.fit(disp=0)

print(model_fit.summary())
# Plot residual errors

residuals = pd.DataFrame(model_fit.resid)

fig, ax = plt.subplots(1,2)

residuals.plot(title="Residuals", ax=ax[0])

residuals.plot(kind='kde', title='Density', ax=ax[1])

plt.show()
model_fit.plot_predict(dynamic=False)

plt.show()
from statsmodels.tsa.stattools import acf



# Create Training and Test

train = d.Confirmed[:40]

test = d.Confirmed[40:]



model = ARIMA(train, order=(4, 2, 1))  

fitted = model.fit(disp=-1)  



# Forecast

fc, se, conf = fitted.forecast(10, alpha=0.05)  # 95% conf



# Make as pandas series

fc_series = pd.Series(fc, index=test.index)

lower_series = pd.Series(conf[:, 0], index=test.index)

upper_series = pd.Series(conf[:, 1], index=test.index)



# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()
model = ARIMA(train, order=(4, 1, 1))  

fitted = model.fit(disp=-1)  

print(fitted.summary())
model = ARIMA(train, order=(2, 1, 1))  

fitted = model.fit(disp=-1)  

print(fitted.summary())
model = ARIMA(train, order=(1, 1, 1))  

fitted = model.fit(disp=-1)  

print(fitted.summary())
d = pd.DataFrame(y).reset_index()
for p in range(1,10):

    for d in range(1,3):

        for q in range(1, 3):

            print(p, d, q)

            print(ARIMA(train, order=(p, d,1)).fit().aic)
d = y
from statsmodels.tsa.stattools import acf



# Create Training and Test

train = d.Confirmed[:40]

test = d.Confirmed[40:]



model = ARIMA(train, order=(1, 2, 2))  

fitted = model.fit(disp=-1)  



# Forecast

fc, se, conf = fitted.forecast(10, alpha=0.05)  # 95% conf



# Make as pandas series

fc_series = pd.Series(fc, index=test.index)

lower_series = pd.Series(conf[:, 0], index=test.index)

upper_series = pd.Series(conf[:, 1], index=test.index)



# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()
#Prediction of march 12th to march 18th

forecasts = (fitted.forecast(steps=7)[0])
#plot the predictions

dates = d['Date'].tolist()+['2020-03-12','2020-03-13','2020-03-14','2020-03-15','2020-03-16','2020-03-17','2020-03-18']

count = d['Confirmed'].tolist()+forecasts.tolist()
final_forecast = pd.DataFrame({'Date': dates,

                             'Confirmed': count

                              })
final_forecast.Date
import seaborn as sns

final_forecast['Date'] = final_forecast['Date'].astype('datetime64')

final_forecast['Date'] = final_forecast['Date'].dt.date

# final_forecast['Date']

# clrs = []

# # clrs = ['red' if x in ['2020-03-12','2020-03-13','2020-03-14','2020-03-15','2020-03-16','2020-03-17','2020-03-18'] for x in final_forecast['Date'].tolist()]



ax = sns.barplot(x='Date', y= 'Confirmed', data=final_forecast, palette=clrs)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.show()
final_forecast['Date']
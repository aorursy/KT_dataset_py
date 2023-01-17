# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install pmdarima

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

import seaborn as sns

from pmdarima import auto_arima

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_pacf,plot_acf

from statsmodels.tsa.seasonal import seasonal_decompose

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/us-candy-production-by-month/candy_production.csv',index_col='observation_date',parse_dates=True)
data.head()
sns.set(rc={'figure.figsize':(20,5)})

data.plot()
data.index
data.index.freq = pd.infer_freq(data.index)
data.index
result = seasonal_decompose(data)

sns.set(rc={'figure.figsize': (20,5)})

result.plot();
plot_acf(data);
plot_pacf(data);
auto_arima(data['IPG3113N'],seasonal=True,m=12,suppress_warnings=True,information_criterion='aic',max_P=5,max_D=5,max_Q=5,max_p=5,max_d=5,max_q=5).summary()
train = data[:500]

test = data[500:]
model = SARIMAX(train['IPG3113N'],order=(3,1,3),seasonal_order=(1,0,2,12))
result_f = model.fit()
pred = result_f.predict(start=len(train),end=len(train)+len(test),type='levels')
fig, ax = plt.subplots()

ax=test.plot(color='red',ax=ax)

ax=pred.plot(color='green',ax=ax)

ax.legend(['test','pred'])
model_forecast = SARIMAX(data,order=(3,1,3),seasonal_order=(1,0,2,12))



model_forecast_fit = model_forecast.fit()



pred_forecast = model_forecast_fit.predict(len(data),len(data)+48,type='levels')
fig, ax = plt.subplots()

ax=data.plot(color='red',ax=ax)

ax=pred_forecast.plot(color='green',ax=ax)

ax.legend(['Original_data','Forecast'])
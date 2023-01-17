# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv('https://raw.githubusercontent.com/krishnaik06/ARIMA-And-Seasonal-ARIMA/master/perrin-freres-monthly-champagne-.csv')
df
df.tail()
df.columns=['Month','Sales']
df.head()
df.dropna(axis=0,inplace=True)
df
df['Month']=pd.to_datetime(df['Month'])
df
df.set_index('Month',inplace=True)
df
df.plot()
from statsmodels.tsa.stattools import adfuller

test_result=adfuller(df['Sales'])
def adfuller_test(Sales):
    result=adfuller(Sales)
    labels=['ADF test statistic','p-value','Lags_used','number of observation used']
    for value,label in zip(result,labels):
        print(label+':'+str(value))
    if result[1]<=0.05:
        print('strong evidence against the null hypothesis(ho) reject the null hypothesis it is stationary')
    else:
        print('it is non stationary')
adfuller_test(df['Sales'])
df['Sales'].shift(1)
df['sales_first_difference']=df['Sales']-df['Sales'].shift(1)
df['seasonal first difference']=df['Sales']-df['Sales'].shift(12)
df
adfuller_test(df['seasonal first difference'].dropna())
df['seasonal first difference'].plot()
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Sales'])
plt.show()
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig=plt.figure(figsize=(12,6))
ax1=fig.add_subplot(211)
fig=plot_acf(df['seasonal first difference'].iloc[13:],lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=plot_pacf(df['seasonal first difference'].iloc[13:],lags=40,ax=ax2)
#for non seasonal data
#p=1,d=1q=0or1
from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(df['Sales'],order=(1,1,1))
model_fit=model.fit()
model_fit.summary()
df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot()
import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1,1,1),seasonal_order=(1,1,1,12))
results=model.fit()
df['forecast']=results.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot()
df.index[-1]
from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+DateOffset(months=x)for x in range(0,24)]
future_dataset_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)
future_dataset_df.tail()
future_df=pd.concat([df,future_dataset_df])
future_df['forecast']=results.predict(start=104,end=120,dynamics=True)
future_df[['Sales','forecast']].plot()

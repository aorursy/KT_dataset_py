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
df_defects = pd.read_csv("../input/monthly-defect-data-simulated/software_defects_simulated.csv")
df_defects['created_date'] = pd.to_datetime(df_defects['created_date'], format="%Y-%m-%d")
df_defects.set_index('created_date', inplace=True)
df_defects = df_defects.groupby(pd.Grouper(freq='M')).sum()


df_defects.head()
df_defects.index

df_defects.isnull().sum()
df_defects.plot(figsize=(12, 6), title="Monthly Defect", c="orange")

from statsmodels.tsa.filters.hp_filter import hpfilter
hp_cycle, hp_trend = hpfilter(df_defects['count'], lamb=14400)
df_defects['trend'] = hp_trend
df_defects[['count', 'trend']].plot(legend=True, figsize=(12, 5))
df_defects.drop('trend', inplace=True, axis=1)
from statsmodels.tsa.seasonal import seasonal_decompose
results = seasonal_decompose(df_defects['count'], model='additive', period=12)
results.plot();
from statsmodels.tsa.stattools import adfuller
def ad_fuller_test(var):
    results_stats = adfuller(var)
    print('ADF Statistic: %f' % results_stats[0])
    print('p-value: %f' % results_stats[1])
    print('Critical Values:')
    for key, value in results_stats[4].items():
        print('\t%s: %.3f' % (key, value))
ad_fuller_test(df_defects['count'])
from statsmodels.tsa.stattools import acf
acf(df_defects['count'])
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df_defects, lags=12);
from pandas.plotting import lag_plot
lag_plot(df_defects['count'])
from statsmodels.graphics.tsaplots import month_plot
month_plot(df_defects['count']);
df_defects.shape
#split the dataset in train and test
train = df_defects.iloc[:40]
test = df_defects.iloc[40:]
start = len(train)
end = len(train) + len(test)-1
!pip install pmdarima
from pmdarima import  auto_arima
stepwise_fit = auto_arima(train['count'],start_p=0,start_q=0,max_p=6,max_q=6,seasonal=True,trace=True,m=15,
                         start_P=0,start_Q=0,max_P=6,max_Q=6,D=1,stepwise=True)
stepwise_fit.summary()
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse

model_sarimax = SARIMAX(train['count'],order=(0,1,0),seasonal_order=(0,1,0,15))
model_sarimax_fit = model_sarimax.fit()
pred_sarimax = model_sarimax_fit.predict(start,end,typ='levels').rename('SARIMAX Predictions')
test.plot(figsize=(12,8),legend=True)
pred_sarimax.plot(legend=True)
error = rmse(test['count'],pred_sarimax)
print("Test Mean",test.mean())
print("SARIMAX Predictions Mean",pred_sarimax.mean())
print("SARIMAX Predictions Error",error)
#Forecast into the future - Monthly forecast
model_fr = SARIMAX(df_defects['count'],order=(0,1,0),seasonal_order=(0,1,0,15))
results = model_fr.fit()
forecast = results.predict(len(df_defects),len(df_defects)+16,typ='levels').rename('SARIMAX Forecast')
df_defects['count'].plot(figsize=(12,8),legend=True)
forecast.plot(legend=True)
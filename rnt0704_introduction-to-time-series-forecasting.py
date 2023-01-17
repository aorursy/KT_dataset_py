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
#importing various libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns

# metrics
from sklearn.metrics import mean_squared_error

# forecasting model
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

import statsmodels.api as sm

from IPython.display import display, HTML

import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline

#loading datasets
train = pd.read_csv(r'/kaggle/input/timeseries/Train.csv')
test = pd.read_csv(r'/kaggle/input/timeseries/Test.csv')
#sample_submission = pd.read_csv(r'C:\Users\Dell\Desktop\Machine learning\analytics vidya\time series\sample_submission.csv')

train.head()
train_original=train.copy()
test_original=test.copy()

test.head()
train.dtypes
train.isnull().sum()
train['Datetime']=pd.to_datetime(train['Datetime'])
train.dtypes
train_original['Datetime']=pd.to_datetime(train_original.Datetime, format='%d-%m-%Y %H:%M')
test_original['Datetime']=pd.to_datetime(test_original.Datetime, format='%d-%m-%Y %H:%M')

test.dtypes
test['Datetime']=pd.to_datetime(test['Datetime'])
test.dtypes
train.index
t = train['Count']
t.head()
# extract datetime to year, month, day, hour
for i in (train, test, train_original, test_original):
  i['year']=i.Datetime.dt.year
  i['month']=i.Datetime.dt.month
  i['day']=i.Datetime.dt.day
  i['Hour']=i.Datetime.dt.hour

train.head()

plt.figure(figsize=(15,8))
plt.plot(train['Count'], label = 'Train')
plt.legend(loc='best')
print('Dataset Columns')
print(train.columns, '\n\n', test.columns)
print('-'*20)
print('Dtypes')
print(train.dtypes,'\n\n', test.dtypes)
print('-'*20)
print(train.shape, '\n\n', test.shape)

train=train_original
test=test_original
#  Work on daily time series
# convert test to daily mean
test.index=test.Datetime
test=test.resample('D').mean()

test.head()
# convert train to daily mean
train.index=train.Datetime
train=train.resample('D').mean()

train.head()
t = train['Count']
t['2014']
plt.figure(figsize=(15,8))

# to make more stationary, trend will be removed
Train_log = np.log(train['Count']) 
moving_avg = Train_log.rolling(24).mean()
plt.plot(Train_log, label='Train log') 
plt.plot(moving_avg, color = 'red', label='Moving avg') 
plt.legend(loc='best')
#Dickey Fuller test

from statsmodels.tsa.stattools import adfuller 
def test_stationarity(timeseries):
  #Determing rolling statistics
  rolmean = timeseries.rolling(24).mean() # 24 hours on each day
  rolstd = timeseries.rolling(24).std()

  #Plot rolling statistics:
  orig = plt.plot(timeseries, color='blue',label='Original')
  mean = plt.plot(rolmean, color='red', label='Rolling Mean')
  std = plt.plot(rolstd, color='black', label = 'Rolling Std')
  plt.legend(loc='best')
  plt.title('Rolling Mean & Standard Deviation')
  plt.show(block=False)

  #Perform Dickey-Fuller test:
  print ('Results of Dickey-Fuller Test:')
  dftest = adfuller(timeseries, autolag='AIC')
  dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

  for key,value in dftest[4].items():
      dfoutput['Critical Value (%s)'%key] = value
  print (dfoutput)
test_stationarity(Train_log.dropna())
train_log_diff = Train_log - Train_log.shift(1) 
test_stationarity(train_log_diff.dropna())
train_log_diff_diff = train_log_diff - train_log_diff.shift(1)
test_stationarity(train_log_diff_diff.dropna())
import statsmodels.api as sm
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
sm.graphics.tsa.plot_acf(train_log_diff_diff.dropna())


sm.graphics.tsa.plot_pacf(train_log_diff.dropna())
# AR model
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(Train_log, order=(2, 1, 0))  # here the q value is zero since it is just the AR model 
results_AR = model.fit(disp=-1)  
plt.plot(train_log_diff.dropna(), label='original') 
plt.plot(results_AR.fittedvalues, color='red', label='predictions') 
plt.legend(loc='best')

import statsmodels.api as sm

fit1 = sm.tsa.statespace.SARIMAX(train_original.Count, order=(2, 1, 4),seasonal_order=(0,1,1,24)).fit() 
pred = fit1.predict(start="2014-6-25", end="2014-9-25", dynamic=True) 
plt.figure(figsize=(16,8)) 
plt.plot( train_original['Count'], label='Train') 
plt.plot(pred, label='SARIMA') 
plt.legend(loc='best') 
plt.show()


pred = fit1.predict(start=test_original.index[0], end=test_original.index[-1], dynamic=True) 
plt.figure(figsize=(16,8)) 
plt.plot( train_original['Count'], label='Train') 
plt.plot(pred, label='SARIMA') 
plt.legend(loc='best') 
plt.show()

submit=test_original
submit['Count']=pred.values
submit.index=submit.ID
submit.drop(['ID','Datetime','year','month','day','Hour'], axis=1, inplace=True)
submit.to_csv('submit.csv')


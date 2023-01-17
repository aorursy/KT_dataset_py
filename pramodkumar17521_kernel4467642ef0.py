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
filepath='../input/for-simple-exercises-time-series-forecasting/Alcohol_Sales.csv'

data=pd.read_csv(filepath, index_col='DATE', parse_dates=True)

data['sales']=data['S4248SM144NCEN']

data.head()
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style='darkgrid')

%matplotlib inline

data.tail()
data['sales'].plot(figsize=(16,8))
data['date']=data.index

data['month']=data['date'].dt.month

data['year']=data['date'].dt.year

data.drop('date', axis=1, inplace=True)

data.head()
plt.figure(figsize=(16,8))

data.groupby('year')['sales'].mean().plot.bar()

plt.show()
plt.figure(figsize=(16,8))

data.groupby('month')['sales'].mean().plot.bar()

plt.show()


data.groupby(['year','month'])['sales'].mean().plot(figsize=(16,8))

plt.show()
monthly=data.resample('m').mean()

yearly=data.resample('y').mean()
fig, axs= plt.subplots(2,1)

monthly['sales'].plot(figsize=(16,8), title='Monthly',fontsize=12, ax=axs[0])

yearly['sales'].plot(figsize=(16,8), title='Yearly', fontsize=12,  ax=axs[1])

fig.tight_layout()

plt.show()
data.shape
train=data.loc[:'2011-12-01']

valid=data.loc['2011-12-01':]
train.shape, valid.shape
train.head()
plt.figure(figsize=(16,8))

train['sales'].plot(kind='line',color='blue',label='train')

valid['sales'].plot(kind='line', color='orange',label='valid')

plt.legend(loc='best')

plt.show()
y_pred = valid.copy()

y_pred['movig_avg']=valid['sales'].rolling(6).mean().iloc[-1]

train['sales'].plot(figsize=(16,8), color='blue', label='Train')

valid['sales'].plot(figsize=(16,8), color='orange', label='Valid')

y_pred['movig_avg'].plot(figsize=(16,8), color='green', label='Moving Averages')

plt.legend(loc='best')

plt.show()
from sklearn.metrics import mean_squared_error 

from math import sqrt 

rms = sqrt(mean_squared_error(valid.sales, y_pred['movig_avg'])) 

print(rms)
import statsmodels.api as sm

sm.tsa.seasonal_decompose(train['sales']).plot() 

result = sm.tsa.stattools.adfuller(train.sales)

fig.tight_layout()

plt.show()
print(result)
from statsmodels.tsa.api import Holt

pred_y=valid.copy()

fit1= Holt(np.asarray(train['sales'])).fit(smoothing_level= 0.3, smoothing_slope= 0.01)

pred_y['holt']=fit1.forecast(len(valid))
plt.figure(figsize=(16,8))

train['sales'].plot(color='blue', label='Train')

valid['sales'].plot(color='orange', label='valid')

pred_y['holt'].plot(color='green', label='Holt')

plt.legend(loc='best')

plt.show()
np.sqrt(mean_squared_error(valid.sales, pred_y['holt']))
from statsmodels.tsa.stattools import adfuller
def test_stationarity(data):

    rolmean=data.rolling(window=12).mean()

    rolstd=data.rolling(window=12).std()

    plt.plot(data, color='blue', label='original')

    plt.plot(rolmean, color='red', label='Rolling Mean')

    plt.plot(rolstd, color='black', label='Rolling std')

    plt.legend(loc='best')

    plt.show()

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(data, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)
from matplotlib.pylab import rcParams 

rcParams['figure.figsize'] = 20,10

test_stationarity(train['sales'])
train_log=np.log(train['sales'])

plt.figure(figsize=(16,8))

mov_avg=train_log.rolling(12).mean()

plt.plot(train_log,label='log')

plt.plot(mov_avg,label='mov_avg')

plt.legend(loc='best')

plt.show()
train_log_mov_avg_diff=train_log-mov_avg

train_log_mov_avg_diff.dropna(inplace=True)

test_stationarity(train_log_mov_avg_diff)
#Train Log diffrence

train_log_diff= train_log - train_log.shift(1)

train_log_diff.fillna(0,inplace=True)

test_stationarity(train_log_diff)
from statsmodels.tsa.stattools import acf, pacf 

lag_acf = acf(train_log_diff.dropna(), nlags=25) 

lag_pacf = pacf(train_log_diff.dropna(), nlags=25, method='ols')
plt.plot(lag_acf) 

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')

plt.title('Autocorrelation Function') 

plt.show() 

plt.plot(lag_pacf) 

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 

plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 

plt.title('Partial Autocorrelation Function') 

plt.show()
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(train_log, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(train_log_diff.dropna(),  label='original') 

plt.plot(results_ARIMA.fittedvalues, color='red', label='predicted') 

plt.legend(loc='best') 

plt.show()
def check_prediction_diff(predict_diff, given_set):

    predict_diff= predict_diff.cumsum().shift().fillna(0)

    predict_base = pd.Series(np.ones(given_set.shape[0]) * np.log(given_set['sales'])[0], index = given_set.index)

    predict_log = predict_base.add(predict_diff,fill_value=0)

    predict = np.exp(predict_log)



    plt.plot(given_set['sales'], label = "Given set")

    plt.plot(predict, color = 'red', label = "Predict")

    plt.legend(loc= 'best')

    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['sales']))/given_set.shape[0]))

    plt.show()
ARIMA_predict_diff=results_ARIMA.predict(start="2011-12-01", end="2019-01-01")

check_prediction_diff(ARIMA_predict_diff, valid)
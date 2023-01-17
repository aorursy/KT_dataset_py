import pandas as pd
df=pd.read_csv('../input/predice-el-futuro/train_csv.csv')
test=pd.read_csv('../input/predice-el-futuro/test_csv.csv')
df.head()
import  matplotlib.pyplot  as       plt
df.plot()

df.columns
df[ 'id']
print(df.describe())
#Set up helper function for data visualization 
def plt_(dataset, title):    
    plt.figure(figsize=(12,6))
    plt.plot(dataset, color = 'b')
    plt.ylabel('Tonnes')
    plt.title(title)
    plt.show()
    
def density_plt_(dataset):
    plt.figure(figsize=(10,5))
    sns.distplot(dataset)
    plt.title('Density plot')
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

import seaborn as sns
import matplotlib.pyplot as plt
sns.pointplot(x= 'id', y='feature', data=df)
plt.show()
#Set up helper function for data visualization 
def plt_(dataset, title):    
    plt.figure(figsize=(12,6))
    plt.plot(dataset, color = 'b')
    plt.ylabel('Tonnes')
    plt.title(title)
    plt.show()
    
def density_plt_(dataset):
    plt.figure(figsize=(10,5))
    sns.distplot(dataset)
    plt.title('Density plot')
    plt.show()

df1 = df.copy()
import pandas                          as      pd
import numpy                           as      np
import matplotlib.pyplot               as      plt
import statsmodels.tsa.holtwinters     as      ets
import statsmodels.tools.eval_measures as      fa
from   sklearn.metrics                 import  mean_squared_error
from   statsmodels.tsa.holtwinters     import  SimpleExpSmoothing, Holt
import math
df1['moving_avg_forecast_4']  = df[ 'feature'].rolling(4).mean()
df1['moving_avg_forecast_6']  = df[ 'feature'].rolling(6).mean()
df1['moving_avg_forecast_8']  = df[ 'feature'].rolling(8).mean()
df1['moving_avg_forecast_12'] = df[ 'feature'].rolling(12).mean()
test.columns

def MAPE(y, yhat): 
    y, yhat = np.array(y), np.array(yhat)
    try:
        mape =  round(np.sum(np.abs(yhat - y)) / np.sum(y) * 100,2)
    except:
        print("Observed values are empty")
        mape = np.nan
    return mape
import pandas                          as      pd
import numpy                           as      np
import matplotlib.pyplot               as      plt
import statsmodels.tsa.holtwinters     as      ets
import statsmodels.tools.eval_measures as      fa
from   sklearn.metrics                 import  mean_squared_error
from   statsmodels.tsa.holtwinters     import  SimpleExpSmoothing

def MAPE(y, yhat): 
    y, yhat = np.array(y), np.array(yhat)
    try:
        mape =  round(np.sum(np.abs(yhat - y)) / np.sum(y) * 100,2)
    except:
        print("Observed values are empty")
        mape = np.nan
    return mape

Petrol              =  pd.read_csv('../input/predice-el-futuro/train_csv.csv')

date_rng            =  pd.date_range(start='1/1/2019', end='30/12/2019', freq='Q')
print(date_rng)

Petrol['TimeIndex'] = pd.DataFrame(date_rng, columns=['Quarter'])
print(Petrol.head(3).T)

plt.plot(Petrol.TimeIndex, Petrol[ 'feature'])
plt.title('Original data before split')
plt.show()

#Creating train and test set 

train             = Petrol[0:int(len(Petrol)*0.7)] 
test              = Petrol[int(len(Petrol)*0.7):]

print("\n Training data start at \n")
print (train[train.TimeIndex == train.TimeIndex.min()],['Year','Quarter'])
print("\n Training data ends at \n")
print (train[train.TimeIndex == train.TimeIndex.max()],['Year','Quarter'])

print("\n Test data start at \n")
print (test[test.TimeIndex == test.TimeIndex.min()],['Year','Quarter'])

print("\n Test data ends at \n")
print (test[test.TimeIndex == test.TimeIndex.max()],['Year','Quarter'])

plt.plot(train.TimeIndex, train[ 'feature'], label = 'Train')
plt.plot(test.TimeIndex, test[ 'feature'],  label = 'Test')
plt.legend(loc = 'best')
plt.title('Original data after split')
plt.show()
# create class
model = SimpleExpSmoothing(np.asarray(train[ 'feature']))
# fit model

alpha_list = [0.1, 0.5, 0.99]

pred_SES  = test.copy() # Have a copy of the test dataset

for alpha_value in alpha_list:

    alpha_str            =  "SES" + str(alpha_value)
    mode_fit_i           =  model.fit(smoothing_level = alpha_value, optimized=False)
    pred_SES[alpha_str]  =  mode_fit_i.forecast(len(test[ 'feature']))
    rmse                 =  np.sqrt(mean_squared_error(test[ 'feature'], pred_SES[alpha_str]))
    mape                 =  MAPE(test[ 'feature'],pred_SES[alpha_str])
###
    print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse, mape))
    plt.figure(figsize=(16,8))
    plt.plot(train.TimeIndex, train[ 'feature'], label ='Train')
    plt.plot(test.TimeIndex, test[ 'feature'], label  ='Test')
    plt.plot(test.TimeIndex, pred_SES[alpha_str], label  = alpha_str)
    plt.title('Simple Exponential Smoothing with alpha ' + str(alpha_value))
    plt.legend(loc='best') 
    plt.show()
import pandas                          as      pd
import numpy                           as      np
import matplotlib.pyplot               as      plt
import statsmodels.tsa.holtwinters     as      ets
import statsmodels.tools.eval_measures as      fa
from   sklearn.metrics                 import  mean_squared_error
from   statsmodels.tsa.holtwinters     import  SimpleExpSmoothing, Holt

def MAPE(y, yhat): 
    y, yhat = np.array(y), np.array(yhat)
    try:
        mape =  round(np.sum(np.abs(yhat - y)) / np.sum(y) * 100,2)
    except:
        print("Observed values are empty")
        mape = np.nan
    return mape

Petrol              = pd.read_csv('../input/predice-el-futuro/train_csv.csv')

date_rng            =  pd.date_range(start='1/1/2019', end='30/12/2019', freq='Q')
print(date_rng)

Petrol['TimeIndex'] = pd.DataFrame(date_rng, columns=['Quarter'])
print(Petrol.head(3).T)

plt.plot(Petrol.TimeIndex, Petrol[ 'feature'])
plt.title('Original data before split')
plt.show()

#Creating train and test set 

train             = Petrol[0:int(len(Petrol)*0.7)] 
test              = Petrol[int(len(Petrol)*0.7):]

print("\n Training data start at \n")
print (train[train.TimeIndex == train.TimeIndex.min()],['Year','Quarter'])
print("\n Training data ends at \n")
print (train[train.TimeIndex == train.TimeIndex.max()],['Year','Quarter'])

print("\n Test data start at \n")
print (test[test.TimeIndex == test.TimeIndex.min()],['Year','Quarter'])

print("\n Test data ends at \n")
print (test[test.TimeIndex == test.TimeIndex.max()],['Year','Quarter'])

plt.plot(train.TimeIndex, train[ 'feature'], label = 'Train')
plt.plot(test.TimeIndex, test[ 'feature'],  label = 'Test')
plt.legend(loc = 'best')
plt.title('Original data after split')
plt.show()
model = Holt(np.asarray(train[ 'feature']))

model_fit = model.fit()

print('')
print('==Holt model Exponential Smoothing Parameters ==')
print('')
alpha_value = np.round(model_fit.params['smoothing_level'], 4)
print('Smoothing Level', alpha_value )
print('Smoothing Slope', np.round(model_fit.params['smoothing_slope'], 4))
print('Initial Level',   np.round(model_fit.params['initial_level'], 4))
print('')

Pred_Holt = test.copy()

Pred_Holt['Opt'] = model_fit.forecast(len(test[ 'feature']))
plt.figure(figsize=(16,8))
plt.plot(train[ 'feature'], label='Train')
plt.plot(test[ 'feature'], label='Test')
plt.plot(Pred_Holt['Opt'], label='HoltOpt')
plt.legend(loc='best')
plt.show()
df_pred_opt          =  pd.DataFrame({'Y_hat':Pred_Holt['Opt'] ,'Y':test[ 'feature'].values})

rmse_opt             =  np.sqrt(mean_squared_error(df_pred_opt.Y, df_pred_opt.Y_hat))
mape_opt             =  MAPE(df_pred_opt.Y, df_pred_opt.Y_hat)

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse_opt, mape_opt))
print(model_fit.params)
import pandas                          as      pd
import numpy                           as      np
import matplotlib.pyplot               as      plt
import statsmodels.tsa.holtwinters     as      ets
import statsmodels.tools.eval_measures as      fa
from   sklearn.metrics                 import  mean_squared_error
from   statsmodels.tsa.holtwinters     import  SimpleExpSmoothing, Holt, ExponentialSmoothing

def MAPE(y, yhat): 
    y, yhat = np.array(y), np.array(yhat)
    try:
        mape =  round(np.sum(np.abs(yhat - y)) / np.sum(y) * 100,2)
    except:
        print("Observed values are empty")
        mape = np.nan
    return mape

Petrol              =  pd.read_csv('../input/predice-el-futuro/train_csv.csv')

date_rng            =  pd.date_range(start='1/1/2019', end='30/12/2019', freq='Q')
print(date_rng)

Petrol['TimeIndex'] = pd.DataFrame(date_rng, columns=['Quarter'])
print(Petrol.head(3).T)

plt.plot(Petrol.TimeIndex, Petrol[ 'feature'])
plt.title('Original data before split')
plt.show()

#Creating train and test set 

train             = Petrol[0:int(len(Petrol)*0.7)] 
test              = Petrol[int(len(Petrol)*0.7):]

print("\n Training data start at \n")
print (train[train.TimeIndex == train.TimeIndex.min()],['Year','Quarter'])
print("\n Training data ends at \n")
print (train[train.TimeIndex == train.TimeIndex.max()],['Year','Quarter'])

print("\n Test data start at \n")
print (test[test.TimeIndex == test.TimeIndex.min()],['Year','Quarter'])

print("\n Test data ends at \n")
print (test[test.TimeIndex == test.TimeIndex.max()],['Year','Quarter'])

plt.plot(train.TimeIndex, train[ 'feature'], label = 'Train')
plt.plot(test.TimeIndex, test[ 'feature'],  label = 'Test')
plt.legend(loc = 'best')
plt.title('Original data after split')
plt.show()
pred1 = ExponentialSmoothing(np.asarray(train[ 'feature']), trend='additive', damped=False, seasonal='additive',
                                  seasonal_periods = 12).fit() #[:'2017-01-01']
print('')
print('== Holt-Winters Additive ETS(A,A,A) Parameters ==')
print('')
alpha_value = np.round(pred1.params['smoothing_level'], 4)
print('Smoothing Level: ', alpha_value)
print('Smoothing Slope: ', np.round(pred1.params['smoothing_slope'], 4))
print('Smoothing Seasonal: ', np.round(pred1.params['smoothing_seasonal'], 4))
print('Initial Level: ', np.round(pred1.params['initial_level'], 4))
print('Initial Slope: ', np.round(pred1.params['initial_slope'], 4))
print('Initial Seasons: ', np.round(pred1.params['initial_seasons'], 4))
print('')

### Forecast for next 16 months


print(pred1.params)
import pandas                          as      pd
import numpy                           as      np
import matplotlib.pyplot               as      plt
import statsmodels.tsa.holtwinters     as      ets
import statsmodels.tools.eval_measures as      fa
from   sklearn.metrics                 import  mean_squared_error
from   statsmodels.tsa.holtwinters     import  SimpleExpSmoothing, Holt, ExponentialSmoothing

def MAPE(y, yhat): 
    y, yhat = np.array(y), np.array(yhat)
    try:
        mape =  round(np.sum(np.abs(yhat - y)) / np.sum(y) * 100,2)
    except:
        print("Observed values are empty")
        mape = np.nan
    return mape

AirPax              = pd.read_csv('../input/predice-el-futuro/train_csv.csv')
date_rng            = pd.date_range(start='1/1/2019', end='31/12/2019', freq='M')
print(date_rng)

AirPax['TimeIndex'] = pd.DataFrame(date_rng, columns=['Month'])
print(AirPax.head())

#Creating train and test set 

train             = AirPax[0:int(len(AirPax)*0.7)] 
test              = AirPax[int(len(AirPax)*0.7):]
print("\n Training data start at \n")
print (train[train.TimeIndex == train.TimeIndex.min()],['Year','Month'])
print("\n Training data ends at \n")
print (train[train.TimeIndex == train.TimeIndex.max()],['Year','Month'])

print("\n Test data start at \n")
print (test[test.TimeIndex == test.TimeIndex.min()],['Year','Month'])

print("\n Test data ends at \n")
print (test[test.TimeIndex == test.TimeIndex.max()],['Year','Month'])

plt.plot(train.TimeIndex, train[ 'feature'], label = 'Train')
plt.plot(test.TimeIndex, test[ 'feature'],  label = 'Test')
plt.legend(loc = 'best')
plt.title('Original data after split')
plt.show()
pred = ExponentialSmoothing(np.asarray(train[ 'feature']),
                                  seasonal_periods=12 ,seasonal='add').fit(optimized=True)

print(pred.params)

print('')
print('== Holt-Winters Additive ETS(A,A,A) Parameters ==')
print('')
alpha_value = np.round(pred.params['smoothing_level'], 4)
print('Smoothing Level: ', alpha_value)
print('Smoothing Slope: ', np.round(pred.params['smoothing_slope'], 4))
print('Smoothing Seasonal: ', np.round(pred.params['smoothing_seasonal'], 4))
print('Initial Level: ', np.round(pred.params['initial_level'], 4))
print('Initial Slope: ', np.round(pred.params['initial_slope'], 4))
print('Initial Seasons: ', np.round(pred.params['initial_seasons'], 4))
print('')

pred_HoltW = test.copy()
pred_HoltW['HoltW'] = model_fit.forecast(len(test[ 'feature']))
plt.figure(figsize=(16,8))
plt.plot(train[ 'feature'], label='Train')
plt.plot(test[ 'feature'], label='Test')
plt.plot(pred_HoltW['HoltW'], label='HoltWinters')
plt.title('Holt-Winters Additive ETS(A,A,A) Parameters:\n  alpha = ' + 
          str(alpha_value) + '  Beta:' + 
          str(np.round(pred.params['smoothing_slope'], 4)) +
          '  Gamma: ' + str(np.round(pred.params['smoothing_seasonal'], 4)))
plt.legend(loc='best')
plt.show()
df_pred_opt =  pd.DataFrame({'Y_hat':pred_HoltW['HoltW'] ,'Y':test[ 'feature'].values})

rmse_opt    =  np.sqrt(mean_squared_error(df_pred_opt.Y, df_pred_opt.Y_hat))
mape_opt    =  MAPE(df_pred_opt.Y, df_pred_opt.Y_hat)

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse_opt, mape_opt))
import pandas                          as      pd
import numpy                           as      np
import matplotlib.pyplot               as      plt
import statsmodels.tsa.holtwinters     as      ets
import statsmodels.tools.eval_measures as      fa
from   sklearn.metrics                 import  mean_squared_error
from   statsmodels.tsa.holtwinters     import  SimpleExpSmoothing, Holt, ExponentialSmoothing

def MAPE(y, yhat): 
    y, yhat = np.array(y), np.array(yhat)
    try:
        mape =  round(np.sum(np.abs(yhat - y)) / np.sum(y) * 100,2)
    except:
        print("Observed values are empty")
        mape = np.nan
    return mape

AirPax              =  pd.read_csv('../input/predice-el-futuro/train_csv.csv')
date_rng            = pd.date_range(start='1/1/2019', end='31/12/2019', freq='M')
print(date_rng)

AirPax['TimeIndex'] = pd.DataFrame(date_rng, columns=['Month'])
print(AirPax.head())

#Creating train and test set 

train             = AirPax[0:int(len(AirPax)*0.7)] 
test              = AirPax[int(len(AirPax)*0.7):]
pred = ExponentialSmoothing(np.asarray(train[ 'feature']),
                                  seasonal_periods=12 ,seasonal='multiplicative').fit(optimized=True)

print(pred.params)

print('')
print('== Holt-Winters Additive ETS(A,A,A) Parameters ==')
print('')
alpha_value = np.round(pred.params['smoothing_level'], 4)
print('Smoothing Level: ', alpha_value)
print('Smoothing Slope: ', np.round(pred.params['smoothing_slope'], 4))
print('Smoothing Seasonal: ', np.round(pred.params['smoothing_seasonal'], 4))
print('Initial Level: ', np.round(pred.params['initial_level'], 4))
print('Initial Slope: ', np.round(pred.params['initial_slope'], 4))
print('Initial Seasons: ', np.round(pred.params['initial_seasons'], 4))
print('')
pred_HoltW = test.copy()

pred_HoltW['HoltWM'] = pred.forecast(len(test[ 'feature']))
plt.figure(figsize=(16,8))
plt.plot(train[ 'feature'], label='Train')
plt.plot(test[ 'feature'], label='Test')
plt.plot(pred_HoltW['HoltWM'], label='HoltWinters')
plt.title('Holt-Winters Multiplicative ETS(A,A,M) Parameters:\n  alpha = ' + 
          str(alpha_value) + '  Beta:' + 
          str(np.round(pred.params['smoothing_slope'], 4)) +
          '  Gamma: ' + str(np.round(pred.params['smoothing_seasonal'], 4)))
plt.legend(loc='best')
plt.show()
df_pred_opt =  pd.DataFrame({'Y_hat':pred_HoltW['HoltWM'] ,'Y':test[ 'feature'].values})

rmse_opt    =  np.sqrt(mean_squared_error(df_pred_opt.Y, df_pred_opt.Y_hat))
mape_opt    =  MAPE(df_pred_opt.Y, df_pred_opt.Y_hat)

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse_opt, mape_opt))
import pandas            as     pd 
import numpy             as     np 
from   sklearn.metrics   import mean_squared_error
from   math              import sqrt
import matplotlib.pyplot as     plt 
import warnings
import datetime as dt
warnings.filterwarnings("ignore")
%matplotlib inline

def MAPE(y, yhat): 
    y, yhat = np.array(y), np.array(yhat)
    try:
        mape =  round(np.sum(np.abs(yhat - y)) / np.sum(y) * 100,2)
    except:
        print("Observed values are empty")
        mape = np.nan
    return mape
#Importing data
def parser(x):
    return pd.datetime.strptime('190'+x, '%Y-%m')
 
df = pd.read_csv('../input/predice-el-futuro/train_csv.csv')

print(df.head())
#df.plot() 

df.Timestamp = pd.to_datetime(df[ 'feature'], format='%Y-%m') 
df.index     = df.Timestamp 
train    =   df[0:int(len(df)*0.7)] 
test     =   df[int(len(df)*0.7):]
### Plot data

train[ 'feature'].plot(figsize=(15,8), title= 's', fontsize=14)
test[ 'feature'].plot(figsize=(15,8), title= 's', fontsize=14)
#Creating train and test set 
train=df1[0:int(len(df1)*0.7)] 
test=df1[int(len(df1)*0.7):]
x_train = train.drop([ 'feature'], axis=1)
x_test  = test.drop([ 'feature'], axis=1)
y_train = train[[ 'feature']]
y_test  = test[[ 'feature']]
from sklearn.linear_model import LinearRegression
model = LinearRegression()
df1 = df.copy()
df1['moving_avg_forecast_4']  = df[ 'feature'].rolling(4).mean()
df1['moving_avg_forecast_6']  = df[ 'feature'].rolling(6).mean()
df1['moving_avg_forecast_8']  = df[ 'feature'].rolling(8).mean()
df1['moving_avg_forecast_12'] = df[ 'feature'].rolling(12).mean()
df=pd.read_csv('../input/predice-el-futuro/train_csv.csv')
#Plotting data 
train.feature.plot(figsize=(15,8), title= 'All day features', fontsize=14) 
test.feature.plot(figsize=(15,8), title= 'All day features', fontsize=14) 
plt.show()
#Naive approach
dd= np.asarray(train.feature) 
y_hat = test.copy() 
y_hat['naive'] = dd[len(dd)-1] 
plt.figure(figsize=(12,8)) 
plt.plot(train.index, train['feature'], label='Train') 
plt.plot(test.index,test['feature'], label='Test') 
plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast') 
plt.legend(loc='best') 
plt.title("Naive Forecast") 
plt.show() 
 
from sklearn.metrics import mean_squared_error 
from math import sqrt 
rms = sqrt(mean_squared_error(test.feature, y_hat.naive)) 
print(rms) 
#simple average
y_hat_avg = test.copy() 
y_hat_avg['avg_forecast'] = train['feature'].mean() 
plt.figure(figsize=(12,8))
plt.plot(train['feature'], label='train') 
plt.plot(test['feature'], label='Test') 
plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
plt.legend(loc='best') 
plt.show() 
 

print(rms) 
#moving average
y_hat_avg = test.copy() 
y_hat_avg['moving_avg_forecast'] = train['feature'].rolling(50).mean().iloc[-1]
plt.figure(figsize=(16,8))
plt.plot(train['feature'], label='Train') 
plt.plot(test['feature'], label='Test') 
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast') 
plt.legend(loc='best')
plt.show()
print("rms: %.4f" % rms)
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 
y_hat_avg = test.copy() 
fit2 = SimpleExpSmoothing(np.asarray(train['feature'])).fit(smoothing_level=0.6,optimized=False) 
y_hat_avg['SES'] = fit2.forecast(len(test))
plt.figure(figsize=(16,8)) 
plt.plot(train['feature'], label='train') 
plt.plot(test['feature'], label='test') 
plt.plot(y_hat_avg['SES'], label='SES') 
plt.legend(loc='best') 
plt.show() 
 
rms = sqrt(mean_squared_error(test.feature, y_hat_avg.SES)) 
print(rms) 
import statsmodels
print(statsmodels.__version__)
import statsmodels.api as sm 
sm.tsa.seasonal_decompose(train.feature, freq=3).plot() 
result = sm.tsa.stattools.adfuller(train.feature)
plt.show()

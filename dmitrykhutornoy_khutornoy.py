import pandas as pd 

import datetime

import numpy as np 

import matplotlib.pyplot as plt 

import statsmodels.api as sm

from pylab import rcParams

import warnings

from pandas.core.nanops import nanmean as pd_nanmean

import xgboost as xgb





from sklearn.metrics import mean_absolute_error



warnings.filterwarnings('ignore')

%matplotlib inline
df = pd.read_csv('../input/sputnik/train.csv')

df['epoch'] = pd.to_datetime(df.epoch)

df.index = df.epoch

df.drop('epoch', axis = 1, inplace = True)

df.head()
train = df.loc[df['type'] == 'train']
train['error']  = np.linalg.norm(train[['x', 'y', 'z']].values - train[['x_sim', 'y_sim', 'z_sim']].values, axis=1)
test = df.loc[df['type'] == 'test']
train.describe()
train_frames = []

for i in range(600):

    train_frames.append(train.loc[train['sat_id'] == i])
test_frames = []

for i in range(600):

    test_frames.append(test.loc[test['sat_id'] == i])
train_frames[0].x.plot( figsize=(15,6),title= 'Ось x', fontsize=14)
train_frames[0].y.plot( figsize=(15,6),title= 'Ось y', fontsize=14)
train_frames[0].z.plot( figsize=(15,6),title= 'Ось z', fontsize=14)
from scipy import signal

seasons=[]

for i in range(600):

    x = np.asarray(train_frames[i].x)

    fs = nfft = nperseg = len(x)

    _, psd = signal.welch(x, fs, nperseg=nperseg, nfft=nfft, window='boxcar', noverlap=0)

    seasons.append(psd.argmax()) 
seasons[:10]
season=24
rcParams['figure.figsize'] = 12, 7

result = sm.tsa.seasonal_decompose(train_frames[0].x,freq=season)

result.plot()

plt.show()
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries, window = 24, cutoff = 0.05):



    #Determing rolling statistics

    rolmean = timeseries.rolling(window).mean()

    rolstd = timeseries.rolling(window).std()



    #Plot rolling statistics:

    fig = plt.figure(figsize=(12, 4))

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show()

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    

    plt.show()

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries.values,autolag='AIC' )

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    pvalue = dftest[1]

    if pvalue < cutoff:

        print('p-value = %.4f. The series is likely stationary.' % pvalue)

    else:

        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)

    

    print(dfoutput)
sample_data=train_frames[0].x
test_stationarity(sample_data)
rcParams['figure.figsize'] = 12, 7

sample_data_diff = sample_data - sample_data.shift(1)

sample_data_diff.dropna(inplace = True)

test_stationarity(sample_data_diff, window = 48)
d=1

D=0
import statsmodels.api as sm

fig, ax = plt.subplots(figsize=(20,8))

sm.graphics.tsa.plot_pacf(sample_data_diff.values, lags=48,ax = ax)

plt.show()
import statsmodels.api as sm

fig, ax = plt.subplots(figsize=(20,8))

sm.graphics.tsa.plot_acf(sample_data_diff.values, lags=50,ax = ax)

plt.show()
# s = 24

# d = 1

# D = 1

# p = 1

# P = 2

# q = 1

# Q = 0

# result_Sar = []

# for i in range(600):

#     sample_data=train_frames[i].error[-300:]

#     try:

#         best_model=sm.tsa.statespace.SARIMAX(sample_data.squeeze(), order=(p, d, q), seasonal_order=(P, D, Q, s)).fit()

#     except Exception:

#         try:

#             best_model=sm.tsa.statespace.SARIMAX(sample_data.squeeze(), order=(p, d, q), seasonal_order=(1, D, Q, s)).fit()

#         except Exception:

#             best_model=sm.tsa.statespace.SARIMAX(sample_data.squeeze(), order=(p, d, q), seasonal_order=(1, 0, Q, s)).fit()

#     pred = best_model.predict(start = sample_data.shape[0], end = sample_data.shape[0]+test_frames[i].shape[0]-1)

#     result_Sar.extend(list(pred))
# pred_id = list(test.id)

# df_pred_err = pd.DataFrame()

# df_pred_err ['id'] = pred_id

# df_pred_err['error'] = result_Sar
# df_pred_err.to_csv('prediction_Sarima.csv',sep=",",index = False)
from scipy.stats import linregress
def prediction(train,test,period = 24):

    count = test.shape[0]//period

    left = test.shape[0]%period

    trend = train.error[-47:].rolling(window=24).mean()

    os1 = (train.iloc[-period:].error.values).tolist()*count

    if left != 0:

        os1 += train.iloc[-period:-(period-left)].error.values.tolist()

    seas = np.array(os1)

    a0,*_  = linregress(np.arange(24),trend.dropna().values)

    seas+=np.array(range(seas.shape[0])*a0)

    return seas
pr = prediction(train_frames[0],test_frames[0])
y = np.hstack((np.asarray(train_frames[0].error), pr))
x = np.hstack((np.asarray(train_frames[0].index), (np.asarray(test_frames[0].index))))
fig,ax = plt.subplots(figsize = (13,9))

plt.plot(x,y)
result_errors = []

for i in range(600):

    pr_err = prediction(train_frames[i],test_frames[i])

    result_errors.extend(list(pr_err))

sz = len(result_errors)
pred_id = list(test.id)
df_pred_err = pd.DataFrame()

df_pred_err ['id'] = pred_id

df_pred_err['error'] = result_errors
df_pred_err
df_pred_err.to_csv('prediction.csv',sep=",",index = False)
#Import necessary libraries
from statsmodels.tsa.stattools import adfuller

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
#import luminol
#from luminol.anomaly_detector import AnomalyDetector
% matplotlib inline
#Reading data set
import os
print(os.listdir('../input'))
df = pd.read_csv('../input/data.csv')
#Exploratory analysis
df.info()    
df.head()
df.describe()
#Data preprocessing
df.sample(3)
ts = df.set_index('date')
#visualization parameters
plt.figure(figsize=(19, 10))
plt.plot(ts)
plt.title('Time Series distribution')
plt.xlabel('Years')
plt.ylabel('Values')
plt.grid()

plt.xticks(rotation=90)
plt.locator_params(nbins=60, axis = 'x')
#Dickey-Fuller Test
def test_stationarity(x):
    
    #Determing rolling statistics
    rolmean = x.rolling(window=2, center=False).mean()
    rolstd = x.rolling(window=2, center=False).std()

    #Plot rolling statistics:
    plt.figure (figsize=(17, 12))
    orig = plt.plot(x, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.grid()
    plt.xticks(rotation=90)
    plt.locator_params(nbins=60, axis = 'x')
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    tstest = adfuller(x['value'], autolag='AIC')
    tsoutput = pd.Series(tstest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in tstest[4].items():
        tsoutput['Critical Value (%s)' % key] = value
    print(tsoutput)
test_stationarity(ts)
#Data normalization and DF test
ts_log = np.log(ts)
test_stationarity(ts_log)
ts_log.head()
#Decomposition
decomposition = seasonal_decompose(np.asarray(ts['value']), freq = 7)

plt.plot(decomposition.trend)
plt.plot(decomposition.seasonal)
plt.plot(decomposition.resid)


decomposition.plot()
plt.show()
#ARIMA
#exploration of PACE and AC functions for the test dataset.

plt.figure(figsize=(17, 5))
lag_acf = acf(ts, nlags=20)
lag_pacf = pacf(ts, nlags=20, method='ols')

plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
#Luminol - only if module 'luminol was installed'
#data preprocessing for the framework
"""data = np.array(ts['value'])
ts_s = pd.Series(data)
ts_dict = ts_s.to_dict()
ts_dict

detector = luminol.anomaly_detector.AnomalyDetector(ts_dict)
anomalies = detector.get_anomalies()

score = detector.get_all_scores()
for timestamp, value in score.iteritems():
    print(timestamp, value)
    """
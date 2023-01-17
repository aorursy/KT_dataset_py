import numpy as np 

import pandas as pd

import datetime

import os

import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#define a parsing function for the timestamps field in int 

def timeparse (time_in_secs):    

    return datetime.datetime.fromtimestamp(float(time_in_secs))
data = pd.read_csv("../input/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv", 

                   parse_dates = True, index_col = [0], date_parser = timeparse)
np.mean(data)
data.describe()
data.plot(subplots = True)
data["Diff"] = data["High"] - data["Low"]

data["Diff"].plot()
data[data['Diff'] > 200]
data[(data.index > '2016-06-23 12:30:00') & (data.index < '2016-06-23 13:00:00')]
open_series = data['Open']

open_series.head()
plt.plot(open_series)
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    

    #Determing rolling statistics

    rolmean = timeseries.rolling(window=12,center=False).mean()

    

    rolstd = timeseries.rolling(window=12,center=False).std()



    #Plot rolling statistics:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    """

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)

    """

    

    
test_stationarity(open_series)
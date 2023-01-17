# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/AirQualityUCI_req.csv')
data.head()
#print(type(data))
import matplotlib.pyplot as plt
%matplotlib inline
def get_data(feature, d):
    chosen_data = d[['Date', feature]]
    #print(chosen_data.head())
    chosen_data = chosen_data[(chosen_data[feature]  > 0)]
    #print(chosen_data.head())
    return chosen_data

def plot_data(d, feature):
    X = [i for i in range(len(d))]
    Y = d[feature].values
    #print(X,Y)
    #plt.title('Data with one hour increments.')
    plt.plot(X,Y)
    
CO_data = get_data('CO(GT)', data)
plot_data(CO_data, 'CO(GT)')
#result = CO_data.groupby(np.arange(len(CO_data))//6).sum()
#CO_data['Date'] = pd.DatetimeIndex(CO_data['Date'], dayfirst=True)
#result = CO_data.groupby([d.strftime('%Y-%m-%d') for d in CO_data['Date']]).mean()

CO_data['Date'] = pd.DatetimeIndex(CO_data['Date'], dayfirst=True)
result = CO_data.groupby([d.strftime('%Y-%m-%d') for d in CO_data['Date']]).mean()

#print(type(result))
#result = result.apply( lambda _df : _df.sort_values(by=['Date']) )
print(result.head())
plot_data(result, 'CO(GT)')
#result = result.values
#print(result.shape)
#print(result[0])
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
decompostion = seasonal_decompose(result['CO(GT)'].values, freq= 15)
fig = plt.figure()
fig = decompostion.plot()
fig.set_size_inches(15,8)
def stationary(d, feature,window):
    data = d[feature].dropna(inplace=False).values
    rolling_mean = d[feature].rolling(window).mean()
    rolling_std =  d[feature].rolling(window).std()
    #rolling_mean = pd.rolling_mean(data,window=10)
    #rolling_std = pd.rolling_std(data,window=10)
    
    #plotting
    orig = plt.plot(data, color='black',label='Data')
    mean = plt.plot(rolling_mean, color='green', label='Rolling Mean')
    std = plt.plot(rolling_std, color='red', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    # Diceky-Fuller test
    
    print ('Results of Dickey-Fuller Test:')
    test_results = adfuller(data, autolag='AIC')
    #''' This is direcctly from online.
    dfoutput = pd.Series(test_results[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    #for key,value in dftest[4].items():
    #    dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    #'''
    #print(test_results)
    #for item in test_results:
    #    print(item)
stationary(result, 'CO(GT)', 15)
# remove seasonality
result['seasonal_shift'] = result['CO(GT)'] - result['CO(GT)'].shift(15)
stationary(result, 'seasonal_shift', 15)
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = tsaplots.plot_acf(result['seasonal_shift'], ax=ax1)
ax2 = fig.add_subplot(212)
fig = tsaplots.plot_pacf(result['seasonal_shift'], ax=ax2)
mod = sm.tsa.statespace.SARIMAX( result['CO(GT)'].values[:320], order=(0,0,0),
                                 seasonal_order=(1,1,1,15)
                                 )
#start_params = [0, 0, 1]
results = mod.fit()
print (results.summary())
l  = results.predict(start = 300, end = 360, dynamic= True)  
actual_data = plt.plot(result['CO(GT)'].values, color='black', label='Data')
predicted_data = plt.plot([i for i in range(300,361)], l, color='green', label = 'predictions')
plt.legend(loc='best')
plt.show()
print(len(l[:55]), len(result['CO(GT)'].values[300:]))
result_df_sarimax = pd.DataFrame.from_dict({ "actual_data" : result['CO(GT)'].values[300:], "predictions" : l[:55] })
result_df_sarimax
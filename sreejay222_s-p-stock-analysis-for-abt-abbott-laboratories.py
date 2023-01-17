# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib as mpl 
from matplotlib.finance import candlestick_ohlc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
rawdata = pd.read_csv('../input/all_stocks_5yr.csv')
dataABT = rawdata.loc[rawdata['Name']  == 'ABT']
dataABT.head()
dataABT[dataABT.open.isnull()]
dataABT.dropna(inplace=True)
dataABT[dataABT.open.isnull()].sum()
data = dataABT.set_index('date')
data.head()
fig,ax1 = plt.subplots(figsize=(20, 10))
plt.plot(data[['open','close','high','low']])
plt.show()
date_data = dataABT[['date']]
#print(len(date_data))
open_data = dataABT[['open']]
#print(len(open_data))
close_data = dataABT[['close']]
low_data = dataABT[['low']]
high_data = dataABT[['high']]
volume = dataABT[['volume']]
open_val = np.array(open_data[1220:],dtype= np.float64)
close_val = np.array(close_data[1220:],dtype= np.float64)
low_val = np.array(low_data[1220:],dtype= np.float64)
high_val = np.array(high_data[1220:],dtype= np.float64)
volume_val = np.array(volume[1220:],dtype= np.float64)
date_val = np.array(date_data[1220:])
data_dates = []
for date in date_val[0:]:
    #print(date)
    new_date = dates.datestr2num(date)
    #print(new_date)
    data_dates.append(new_date)

#print(len(data_dtes))
#print(data_dates[2])
i = 0 
ohlc_data = []
while i < len(data_dates):
    x = data_dates[i],open_val[i],high_val[i],low_val[i],close_val[i],volume_val[i]
    ohlc_data.append(x)
    i += 1
dayFormatter = dates.DateFormatter('%d-%b-%Y')
fig,ax1 = plt.subplots(figsize=(20, 10))
candlestick_ohlc(ax1,ohlc_data,width=1.5, colorup = 'g', colordown = 'r', alpha = 0.8)
plt.plot(data_dates,open_val)
plt.plot(data_dates,close_val)
plt.plot(data_dates,low_val)
plt.plot(data_dates,high_val)
ax1.xaxis.set_major_formatter(dayFormatter)
plt.ylabel('Stock Price')
plt.xlabel('Dates')
plt.show()
data[['volume']].iloc[1220:].plot.bar()
plt.show()
from pandas.tools.plotting import lag_plot
lag_plot(data['high'])
plt.show()
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data['high'])
plt.show()
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error 

dataHigh = data['high']
values = DataFrame(dataHigh.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
# split into train and test sets
X = dataframe.values
train, test = X[1:len(X)-30], X[len(X)-30:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
 
# persistence model
def model_persistence(x):
    return x
 
# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
# plot predictions vs expected
reg_val, = plt.plot(predictions,color='r',label=u'Predicted Linear')
true_val, = plt.plot(test_y,color='g', label='True Values')
plt.legend(handles=[true_val,reg_val])
plt.ylabel('Dollars')
plt.xlabel('Days')
plt.show()
from statsmodels.tsa.ar_model import AR
train, test = dataHigh[1:len(dataHigh)-30], dataHigh[len(dataHigh)-30:]
# train autoregression
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
reg_val, = plt.plot(predictions,color='r',label=u'Auto-Regression')
true_val, = plt.plot(test,color='g', label='True Values')
plt.legend(handles=[true_val,reg_val])
plt.ylabel('Dollars')
plt.xlabel('Days')
plt.show()
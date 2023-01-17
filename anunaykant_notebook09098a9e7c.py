# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/amazon-stocks-lifetime-dataset/AMZN.csv")
df.head(10)
df.info()
df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
df.info()

data = pd.DataFrame({'Date':df['Date'],'Closing Price':df['Close']})
data.head(10)
data.index = data.Date
data = data.drop('Date',axis=1)
data.head()
data = data["2016":]
data.head(10)
len(data)
plt.figure(figsize=(10,10))
plt.plot(data.index,data['Closing Price'])
plt.xlabel("date")
plt.ylabel("closing price")

def myplot(series):
    plt.figure(figsize=(10,10))
    plt.plot(data.index,series)
    plt.xlabel("date")
    plt.ylabel("closing price")
    
    

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data ,model = 'additive',period = 20)
Seasonal = result.seasonal.to_numpy()
Trend = result.trend.to_numpy()
data_original = data.to_numpy()
myplot(result.trend)
myplot(result.seasonal)
myplot(result.resid)

data_series = result.resid
data_series.replace([np.inf, -np.inf], np.nan, inplace=True) 
data_series = data_series.fillna(0)
myplot(data_series)

data_series = np.log(300 + data_series)
myplot(data_series)
from statsmodels.tsa.stattools import adfuller
data_series = data_series.fillna(0)
result = adfuller(data_series)
print("The p-value is " + str(result[1]))
if result[1] < 0.05:
    print("data-series is stationary")
else:
    print("data-series is not stationary")

data_array = data_series.to_numpy()
data_array = data_array
train = data_array[0: 900]
test = data_array[900:976]
x = []
val = 0
for c in test:
    x.append(val+900)
    val = val + 1
    
plt.plot(train,label = "train")
plt.plot(x,test,label = "test")
plt.legend()
#Simple moving Average
df_sma = data_series.rolling(window = 5).mean()
plt.plot(df_sma)
data_sma = df_sma.to_numpy()
data_sma = data_sma[900:]
error = mean_squared_error(test,data_sma)
print("The error for exponential moving average is " + str(error))
plt.figure(figsize=(10,10))
plt.plot(data_series.index[900:],test,label="test")
plt.plot(data_series.index[900:],data_sma,label= "predictions")
plt.legend()
#Exponential moving average
df_ema = data_series.ewm(span = 5,adjust = False).mean()
plt.plot(df_ema)
data_ema = df_ema.to_numpy()
data_ema = data_ema[900:]
error = mean_squared_error(test,data_ema)
print("The error for exponential moving average is "+ str(error))
plt.figure(figsize=(10,10))
plt.plot(data_series.index[900:],test,label="test")
plt.plot(data_series.index[900:],data_ema,label= "predictions")
plt.legend()
from statsmodels.tsa.stattools import acf, pacf
plt.plot(acf(train[:50]))
plt.plot(pacf(train))
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

predictions = []
input = []
for x in train:
    input.append(x)
for x in test:
    model = ARIMA(input,order=(2,0,1))
    output = model.fit(disp=0).forecast()
    predictions.append(output[0])
    input.append(output[0])
error = mean_squared_error(test,predictions)
print("The mean squared error is given as " + str(error))
    
plt.figure(figsize=(10,10))    
plt.plot(data_series.index[900:],test,label="test")
plt.plot(data_series.index[900:],predictions,label= "predictions")
plt.legend()
plt.figure(figsize=(10,10))
X = []
A = []
E = []
S = []
from math import exp
for i in range(len(test)):
    A.append(exp(predictions[i]) + Seasonal[i+900] + Trend[i+900])
    X.append(exp(test[i]) + Seasonal[i+900] + Trend[i+900])
    E.append(exp(data_ema[i]) + Seasonal[i+900] + Trend[i+900])
    S.append(exp(data_sma[i]) + Seasonal[i+900] + Trend[i+900])
    
plt.plot(data_series.index[900:],X,label="test")
plt.plot(data_series.index[900:],A,label= "Arima_predictions")
plt.plot(data_series.index[900:],E,label= "exponentialMA_predictions")
plt.plot(data_series.index[900:],S,label= "SimpleMA_predictions")
plt.title("Stock price forecasting using different approaches.")
plt.legend()
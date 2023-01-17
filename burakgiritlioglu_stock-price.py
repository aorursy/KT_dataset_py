import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
from statsmodels.tsa.arima_model import ARIMA


import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')



df=pd.read_csv("/kaggle/input/StockPrediction.csv")
df.columns = ['date', 'price', 'opening', 'dailymax', 'dailymin', 'volume', 'difference']
df.info()
df.isna()
df.isnull()
df
df.isnull().sum()
df.price = df.price.str.replace('.', '')
df.opening = df.opening.str.replace('.', '')
df.dailymax = df.dailymax.str.replace('.', '')
df.dailymin = df.dailymin.str.replace('.', '')

#def func(var):
    
#    var = var.str.replace('.', '')

#dft=[df.price,df.opening,df.dailymax,df.dailymin]
#for n in dft:
#    func(n)

df=df.replace({',': '.'}, regex=True)
df.difference=df.difference.replace({'%': ''}, regex=True)
df.price
from datetime import datetime
df["date"]= pd.to_datetime(df["date"]) 
df.info()
d = { 'M': 1000000, 'B': 1000000000}
df.loc[:, 'volume'] = pd.to_numeric(df['volume'].str[:-1]) * \
    df['volume'].str[-1].replace(d)
df
df.info()
df.price = df.price.astype(float)
df.opening = df.opening.astype(float)
df.dailymax = df.dailymax.astype(float)
df.dailymin = df.dailymin.astype(float)
df.volume = df.volume.astype(float)
df.difference = df.difference.astype(float)
df.info()
df['closing'] = df['opening'] + df['difference']
df
#Visualize the closing price history
df1 = df.groupby('date').aggregate({'price':'mean'})
plt.figure(figsize=(16,8))
sns.lineplot(x = df1.index, y = df1.price, data = df1)
plt.show()
df2 = df.groupby('date').aggregate({'opening':'mean'})
plt.figure(figsize=(16,8))
sns.lineplot(x = df2.index, y = df2.opening, data = df2)
plt.show()
df3 = df.groupby('date').aggregate({'dailymax':'mean'})
plt.figure(figsize=(16,8))
sns.lineplot(x = df3.index, y = df3.dailymax, data = df3)
plt.show()
df4 = df.groupby('date').aggregate({'dailymin':'mean'})
plt.figure(figsize=(16,8))
sns.lineplot(x = df4.index, y = df4.dailymin, data = df4)
plt.show()
df
df = df.set_index('date')
training_set = df[:'2016'].iloc[:,1:2].values
test_set = df['2017':].iloc[:,1:2].values
df["closing"][:'2016'].plot(figsize=(16,4),legend=True)
df["closing"]['2017':].plot(figsize=(16,4),legend=True)
plt.legend(['Training set (Before 2017)','Test set (2017 and beyond)'])
plt.title('Price level over time')
plt.show()
df_final = pd.Series(df['closing'])
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
def check_stationarity(ts_data):
    
    # Rolling statistics
    roll_mean = ts_data.rolling(30).mean()
    roll_std = ts_data.rolling(5).std()
    
    # Plot rolling statistics
    fig = plt.figure(figsize=(20,10))
    plt.subplot(211)
    plt.plot(ts_data, color='black', label='Original Data')
    plt.plot(roll_mean, color='red', label='Rolling Mean(30 days)')
    plt.legend()
    plt.subplot(212)
    plt.plot(roll_std, color='green', label='Rolling Std Dev(5 days)')
    plt.legend()
    
    # Dickey-Fuller test
    print('Dickey-Fuller test results\n')
    df_test = adfuller(ts_data, regresults=False)
    test_result = pd.Series(df_test[0:4], index=['Test Statistic','p-value','# of lags','# of obs'])
    print(test_result)
    for k,v in df_test[4].items():
        print('Critical value at %s: %1.5f' %(k,v))
type(df_final)
check_stationarity(df_final)
df_final_log = np.log(df_final)
df_final_log.head()
df_final_log.dropna(inplace=True)
check_stationarity(df_final_log)
df_final_log_diff = df_final_log - df_final_log.shift()
df_final_log_diff.dropna(inplace=True)
check_stationarity(df_final_log_diff)
df_final_diff = df_final - df_final.shift()
df_final_diff.dropna(inplace=True)
check_stationarity(df_final_diff)
from statsmodels.tsa.stattools import acf, pacf
df_acf = acf(df_final_diff)

df_pacf = pacf(df_final_diff)
import statsmodels.api as sm
fig1 = plt.figure(figsize=(20,10))
ax1 = fig1.add_subplot(211)
fig1 = sm.graphics.tsa.plot_acf(df_acf, ax=ax1)
ax2 = fig1.add_subplot(212)
fig1 = sm.graphics.tsa.plot_pacf(df_pacf, ax=ax2)
model = ARIMA(df_final_diff, (1,1,0))
fit_model = model.fit(full_output=True)
predictions = model.predict(fit_model.params, start=1760, end=1769)
fit_model.summary()
predictions
fit_model.predict(start=1760, end=1769)
pred_model_diff = pd.Series(fit_model.fittedvalues, copy=True)
pred_model_diff.head()
pred_model_diff_cumsum = pred_model_diff.cumsum()
pred_model_diff_cumsum.head()
# Element-wise addition back to original time series
df_final_trans = df_final.add(pred_model_diff_cumsum, fill_value=0)
# Last 5 rows of fitted values
df_final_trans.tail()
plt.figure(figsize=(20,10))
plt.plot(df_final, color='black', label='Original data')
plt.plot(df_final_trans, color='red', label='Fitted Values')
plt.legend()
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8

data_arima = df['closing']

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data_arima[-10000:], model='multiplicative', freq=12)
plt.figure(figsize=(14,8))
fig = result.plot()
plt.show()
history = [x for x in training_set]
y = test_set
# make first prediction
predictions = list()
model = ARIMA(history, order=(1,1,0))
model_fit = model.fit(disp=0)
yhat = model_fit.forecast()[0]
predictions.append(yhat)
history.append(y[0])
# rolling forecasts
for i in range(1, len(y)):
    # predict
    model = ARIMA(history, order=(1,1,0))
    model_fit = model.fit(disp=0)
    yhat = model_fit.forecast()[0]
    # invert transformed prediction
    predictions.append(yhat)
    # observation
    obs = y[i]
    history.append(obs)
# report performance
mse = math.sqrt(mean_squared_error(y, predictions))
print('RMSE: '+str(mse))
mae = mean_absolute_error(y, predictions)
print('MAE: '+str(mae))
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
X_train = []
y_train = []
for i in range(60,2893):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=20, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=20, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=20, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=20))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
# Fitting to the training set
regressor.fit(X_train,y_train,epochs=20,batch_size=64)
# Now to get the test set ready in a similar way as the training set.
# The following has been done so forst 60 entires of test set have 60 previous values which is impossible to get unless we take the whole 
# 'High' attribute data for processing
dataset_total = pd.concat((df["closing"][:'2016'],df["closing"]['2017':]),axis=0)
inputs = dataset_total[len(dataset_total)-len(test_set) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = sc.transform(inputs)
# Preparing X_test and predicting the prices
X_test = []
for i in range(60,712):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
plt.plot(test_set, color='red',label='Price')
plt.plot(predicted_stock_price, color='blue',label='Predicted Price')
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
predicted_stock_price
test_set
#return_rmse(test_set,predicted_stock_price)
#def return_rmse(test,predicted):
mse = math.sqrt(mean_squared_error(test_set, predicted_stock_price))
print('RMSE: '+str(mse))
mae = mean_absolute_error(test_set, predicted_stock_price)
print('MAE: '+str(mae))

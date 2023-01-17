# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,5)
import datetime
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
from sklearn.metrics import mean_squared_error
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/BIIB_data.csv")
df.shape
df.head()
df_stocks = df.iloc[:1000, 3]       #first 300 are to fit the model
df_holdout= df.iloc[1000: , 3]      #validation set
df_stocks.plot()
df_holdout.plot(color='green')
df_stocks_diff = df_stocks-df_stocks.shift()
plt.figure(figsize=(20,8))
plt.title('Differenced Time Series')
df_stocks_diff.plot(color='red')
df_stocks_diff.head()
#first value is NaN so replace it with original time series
df_stocks_diff.iloc[0] = df_stocks.iloc[0]
df_stocks_diff.head()
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(df_stocks, nlags=10)    #for 10 lags
lag_pacf = pacf(df_stocks, nlags=10, method='ols')   #for 10 lags
plt.figure(figsize=(13,6))
plt.subplot(121)
plt.title('ACF')
plt.plot(lag_acf, color='purple')
plt.axhline(y=0, linestyle='--', color = 'red')
plt.axhline(y=-1.96/np.sqrt(len(df_stocks_diff)), linestyle='--', color='red')
plt.axhline(y = 1.96/np.sqrt(len(df_stocks_diff)), linestyle='--', color='red')

plt.subplot(122)
plt.title('PACF')
plt.plot(lag_pacf, color='green')
plt.axhline(y=0, linestyle='--', color='red')
plt.axhline(y=-1.96/np.sqrt(len(df_stocks_diff)), linestyle='--', color='red')
plt.axhline(y = 1.96/np.sqrt(len(df_stocks_diff)), linestyle='--', color='red')
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df_stocks, order=(2,1,1))
results_AR = model.fit(disp=-1)
pred_AR = results_AR.predict(start=len(df_stocks_diff), end=len(df_stocks_diff) + 258)
pred_AR.iloc[0] = pred_AR.iloc[0] + df_stocks.iloc[998]
pred_AR = pred_AR.cumsum()
pred_AR.head()

pred_AR.index = df_holdout.index

plt.plot(df_stocks, color='grey')
plt.plot(df_holdout, color='green')
plt.plot(pred_AR, color='red')
error = mean_squared_error(df_holdout, pred_AR)
print("Current Error of our model is ", error)
df = pd.read_csv("/kaggle/input/BIIB_data.csv")
df.head()
from datetime import datetime
def components(x):
    #function to extract date, month and year as a feature
    date = datetime.strptime(x, '%m/%d/%Y')
    return (date.day,date.month,date.year)

df['Day'] = df['date'].apply(lambda x:components(x)[0])
df['Month'] = df['date'].apply(lambda x:components(x)[1])
df['Year'] = df['date'].apply(lambda x:components(x)[2])
df.head()
#Prepare x and y variable
y = df['close']
df = df.drop(['Name', 'date','close','open','volume','high', 'low'],1)
df.head()
df.shape
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
scale = MinMaxScaler()
df = scale.fit_transform(df)
pd.DataFrame(df).head()
df_train = df[:1100,:]
y_train = np.array(y.iloc[:1100])
df_test = df[1100:,:]
y_test = np.array(y.iloc[1100:])

df_train.shape, df_test.shape
model= Sequential()
model.add(LSTM(250, input_shape=(df_train.shape[1],1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
df_train= df_train.reshape((df_train.shape[0], df_train.shape[1], 1))
df_test = df_test.reshape((df_test.shape[0], df_test.shape[1],1))
model.fit(df_train, y_train, epochs=400, shuffle=False)
pred_lstm = model.predict(df_test)
train_val = pd.DataFrame(y_train, index = range(1100))
holdout_predictions = pd.DataFrame(pred_lstm, index = range(1100, 1100+len(pred_lstm)))
holdout_val = pd.DataFrame(y_test, index = range(1100, 1100+len(pred_lstm)))
plt.title('LSTM Model')
plt.plot(train_val, color='grey')
plt.plot(holdout_val, color ='green')
plt.plot(holdout_predictions, color='red')
error = mean_squared_error(holdout_val, pred_lstm)
print(error)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
param_dict = {'n_estimators':[1,5,10]}
rf = RandomForestRegressor()
cv = GridSearchCV(rf, param_grid=param_dict, scoring='neg_mean_squared_error')
df_train = df[:1100, :]
y_train = np.array(y.iloc[:1100])
df_test = df[1100:, :]
y_test = np.array(y.iloc[1100:])
cv.fit(df_train, y_train)
cv.best_score_
pred_rf = cv.best_estimator_.predict(df_test)
train_val = pd.DataFrame(y_train, index = range(1100))
holdout_pred = pd.DataFrame(pred_rf, index=range(1100, 1100+len(pred_rf)))
holdout_values = pd.DataFrame(y_test, index=range(1100,1100+len(pred_rf)))
plt.plot(train_val, color='grey')
plt.plot(holdout_val, color='green')
plt.plot(holdout_pred, color='red')
error = mean_squared_error(y_test, pred_rf)
print(error)

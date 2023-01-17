# !pip install --upgrade scipy

# !pip install --ignore-installed scipy statsmodels

!pip install pmdarima
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from pmdarima import auto_arima

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error

from statsmodels.tools.eval_measures import rmse
#load the data

data = pd.read_csv('../input/air-passengers/AirPassengers.csv')

data.head()
data.info()
data['Month'] = pd.to_datetime(data.Month)

data.head()
#divide into train and validation set

train = data.loc[:len(data)-12, :] #data[:int(0.7*(len(data)))]

test = data.loc[len(data)-12:, :] #data[int(0.7*(len(data))):]



fig, ax = plt.subplots(figsize=(25, 5))

ax.plot('Month', '#Passengers', data=train, label='Train')

ax.plot('Month', '#Passengers', data=test, label='Test')

ax.legend()

ax.set_title('No of air passengers')
# model = auto_arima(train['#Passengers'], trace=True, error_action='ignore', suppress_warnings=True)

model = auto_arima(train['#Passengers'], 

                   seasonal=True, 

                   m=12,

                   max_p=7, max_d=5, max_q=7, 

                   max_P=4, max_D=4, max_Q=4, 

                   scoring='mse',

                   trace=True, error_action='ignore', suppress_warnings=True)

model.fit(train['#Passengers'])



forecast = test.copy()

forecast['Predicted_Passengers'] = model.predict(n_periods=len(test))



fig, ax = plt.subplots(figsize=(25, 5))

ax.plot(train['Month'], train['#Passengers'], label='Train')

ax.plot(test['Month'], test['#Passengers'], label='Valid')

ax.plot(forecast['Month'], forecast['Predicted_Passengers'], label='Prediction')

fig.autofmt_xdate() # make space for and rotate the x-axis tick labels

ax.legend()

ax.set_title('No of air passengers')
model.summary()
forecast.head()
#calculate rmse

rmse = np.sqrt(mean_squared_error(test['#Passengers'], forecast['Predicted_Passengers']))

print('RMSE: ', rmse)
import numpy as np

import pandas as pd

import os

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 

from statsmodels.tsa.seasonal import seasonal_decompose 

#from pmdarima import auto_arima                        

from sklearn.metrics import mean_squared_error

from statsmodels.tools.eval_measures import rmse

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/monthly-beer-production-in-austr.csv')
df.head()
df.info()
df.Month = pd.to_datetime(df.Month)
df = df.set_index("Month")

df.head()
df.index.freq = 'MS'
plt.figure(figsize=(18,9))

plt.plot(df.index, df["Monthly beer production"], linestyle="-")

plt.xlabel=('Dates')

plt.ylabel=('Total Production')

plt.show();
a = seasonal_decompose(df["Monthly beer production"], model = "add")

a.plot();
import matplotlib.pyplot as plt

plt.figure(figsize = (16,7))

a.seasonal.plot();
#auto_arima(df['Monthly beer production'], seasonal=True, m=12,max_p=7, max_d=5,max_q=7, max_P=4, max_D=4,max_Q=4).summary()
train_data = df[:len(df)-12]

test_data = df[len(df)-12:]
arima_model = SARIMAX(train_data['Monthly beer production'], order = (2,1,1), seasonal_order = (4,0,3,12))

arima_result = arima_model.fit()

arima_result.summary()
arima_pred = arima_result.predict(start = len(train_data), end = len(df)-1, typ="levels").rename("ARIMA Predictions")

arima_pred
test_data['Monthly beer production'].plot(figsize = (16,5), legend=True)

arima_pred.plot(legend = True);
arima_rmse_error = rmse(test_data['Monthly beer production'], arima_pred)

arima_mse_error = arima_rmse_error**2

mean_value = df['Monthly beer production'].mean()



print(f'MSE Error: {arima_mse_error}\nRMSE Error: {arima_rmse_error}\nMean: {mean_value}')
test_data['ARIMA_Predictions'] = arima_pred
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)

scaled_test_data = scaler.transform(test_data)
from keras.preprocessing.sequence import TimeseriesGenerator



n_input = 12

n_features= 1

generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM



lstm_model = Sequential()

lstm_model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))

lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mse')



lstm_model.summary()
lstm_model.fit_generator(generator,epochs=20)
losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))

plt.xticks(np.arange(0,21,1))

plt.plot(range(len(losses_lstm)),losses_lstm);
lstm_predictions_scaled = list()



batch = scaled_train_data[-n_input:]

current_batch = batch.reshape((1, n_input, n_features))



for i in range(len(test_data)):   

    lstm_pred = lstm_model.predict(current_batch)[0]

    lstm_predictions_scaled.append(lstm_pred) 

    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
lstm_predictions_scaled
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
lstm_predictions
test_data['LSTM_Predictions'] = lstm_predictions
test_data
test_data['Monthly beer production'].plot(figsize = (16,5), legend=True)

test_data['LSTM_Predictions'].plot(legend = True);
lstm_rmse_error = rmse(test_data['Monthly beer production'], test_data["LSTM_Predictions"])

lstm_mse_error = lstm_rmse_error**2

mean_value = df['Monthly beer production'].mean()



print(f'MSE Error: {lstm_mse_error}\nRMSE Error: {lstm_rmse_error}\nMean: {mean_value}')
df.info()
df_pr = df.copy()

df_pr = df.reset_index()
df_pr.columns = ['ds','y'] # To use prophet column names should be like that
train_data_pr = df_pr.iloc[:len(df)-12]

test_data_pr = df_pr.iloc[len(df)-12:]
from fbprophet import Prophet
m = Prophet()

m.fit(train_data_pr)

future = m.make_future_dataframe(periods=12,freq='MS')

prophet_pred = m.predict(future)
prophet_pred.tail()
prophet_pred = pd.DataFrame({"Date" : prophet_pred[-12:]['ds'], "Pred" : prophet_pred[-12:]["yhat"]})
prophet_pred = prophet_pred.set_index("Date")
prophet_pred.index.freq = "MS"
prophet_pred
test_data["Prophet_Predictions"] = prophet_pred['Pred'].values
import seaborn as sns
plt.figure(figsize=(16,5))

ax = sns.lineplot(x= test_data.index, y=test_data["Monthly beer production"])

sns.lineplot(x=test_data.index, y = test_data["Prophet_Predictions"]);
prophet_rmse_error = rmse(test_data['Monthly beer production'], test_data["Prophet_Predictions"])

prophet_mse_error = prophet_rmse_error**2

mean_value = df['Monthly beer production'].mean()



print(f'MSE Error: {prophet_mse_error}\nRMSE Error: {prophet_rmse_error}\nMean: {mean_value}')
rmse_errors = [arima_rmse_error, lstm_rmse_error, prophet_rmse_error]

mse_errors = [arima_mse_error, lstm_mse_error, prophet_mse_error]

errors = pd.DataFrame({"Models" : ["ARIMA", "LSTM", "Prophet"],"RMSE Errors" : rmse_errors, "MSE Errors" : mse_errors})
plt.figure(figsize=(16,9))

plt.plot_date(test_data.index, test_data["Monthly beer production"], linestyle="-")

plt.plot_date(test_data.index, test_data["ARIMA_Predictions"], linestyle="-.")

plt.plot_date(test_data.index, test_data["LSTM_Predictions"], linestyle="--")

plt.plot_date(test_data.index, test_data["Prophet_Predictions"], linestyle=":")

plt.legend()

plt.show()
print(f"Mean: {test_data['Monthly beer production'].mean()}")

errors
test_data
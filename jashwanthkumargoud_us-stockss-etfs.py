import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
dataset = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/alxn.us.txt',index_col="Date",parse_dates=True)
dataset.head()
dataset.tail()
dataset.isna().any()
dataset.info()
dataset['Open'].plot(figsize=(16,6))
dataset.rolling(7).mean().head(20)
dataset['Open'].plot(figsize=(16,6))
dataset.rolling(window=30).mean()['Close'].plot()
dataset['Close: 30 Day Mean'] = dataset['Close'].rolling(window=30).mean()
dataset[['Close','Close: 30 Day Mean']].plot(figsize=(16,6))
dataset['Close'].expanding(min_periods=1).mean().plot(figsize=(16,6))
training_set=dataset['Open']
training_set=pd.DataFrame(training_set)


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))


regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


regressor.add(Dense(units = 1))


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


regressor.fit(X_train, y_train, epochs = 30, batch_size = 32)
dataset_test = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/ETFs/drw.us.txt',index_col="Date",parse_dates=True)
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_test.head()
dataset_test.tail()
dataset_test.info()
test_set=dataset_test['Open']
test_set=pd.DataFrame(test_set)
test_set.info()

dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price=pd.DataFrame(predicted_stock_price)
predicted_stock_price.info()

plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price Data')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price Data')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

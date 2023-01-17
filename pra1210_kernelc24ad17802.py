import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
df_train = pd.read_csv('../input/Google_Stock_Price_Train.csv')
training_set = df_train.iloc[:, 1:2].values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
ID_train = []
D_train = []
for i in range(60, 1258):
    ID_train.append(training_set_scaled[i - 60: i, 0])
    D_train.append(training_set_scaled[i, 0])
ID_train, D_train = np.array(ID_train), np.array(D_train)
ID_train = np.reshape(ID_train, (ID_train.shape[0], ID_train.shape[1], 1))
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (ID_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(ID_train, D_train, epochs = 50, batch_size = 32)
df_test = pd.read_csv('../input/Google_Stock_Price_Test.csv')
real_stock_price = df_test.iloc[:, 1:2].values
df_total = pd.concat((df_train['Open'], df_test['Open']), axis = 0)
inputs = df_total[len(df_total) - len(df_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
D_test = []
for i in range(60, 80):
    D_test.append(inputs[i - 60:i, 0])
D_test = np.array(D_test)
D_test = np.reshape(D_test, (D_test.shape[0], D_test.shape[1], 1))
predicted_stock_price = regressor.predict(D_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'REAL GOOGLE STOCK PRICE')
plt.plot(predicted_stock_price, color = 'blue', label = 'PREDICTED GOOGLE STOCK PRICE')
plt.title('GOOGLE STOCK PRICE PREDICTION')
plt.xlabel('TIME')
plt.ylabel('GOOGLE STOCK PRICE')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df_train = pd.read_csv("../input/bitcoin-price/bctrain.csv")
df_processed = df_train.iloc[:, 1:2].values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
df_train_scaled = scaler.fit_transform(df_processed)
features_set = []
labels = []
for i in range(30, 776):
    features_set.append(df_train_scaled[i-30:i, 0])
    labels.append(df_train_scaled[i, 0])
features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['mse'])
model.fit(features_set, labels,epochs = 50, batch_size = 32)
df_test = pd.read_csv("../input/bitcoin-price/bctest.csv")
df_test_processed = df_test.iloc[:, 1:2].values
df_total = pd.concat((df_train['Price'], df_test['Price']), axis=0)
test_inputs = df_total[len(df_total) - len(df_test) - 30:].values
test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)
test_features = []
for i in range(30, 60):
    test_features.append(test_inputs[i-30:i, 0])
test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
predictions = model.predict(test_features)
predictions = scaler.inverse_transform(predictions)
mape=np.mean(np.abs((predictions-df_test_processed) / df_test_processed))*100
mape
plt.figure(figsize=(12,6))
plt.plot(df_test_processed, color='blue', label='Actual Bitcoin Price')
plt.plot(predictions , color='red', label='Predicted Bitcoin Price')
plt.title('Bitcoin Prediction')
plt.xlabel('April Month Date')
plt.ylabel('Price')
plt.legend()
plt.show()


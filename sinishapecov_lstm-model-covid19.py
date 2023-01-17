import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.callbacks import EarlyStopping
import os
%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_mkd = pd.read_csv('/kaggle/input/corona-north-macedonia/corona_north_macedonia.csv');
df_china = pd.read_csv('/kaggle/input/corona-north-macedonia/corona_china.csv')
data = np.array(df_china['active'])
full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(data.reshape(-1,1))
length = 15
batch_size = 1
n_features = 1
generator = TimeseriesGenerator(scaled_full_data,scaled_full_data, length=length, batch_size=1)
model = Sequential()
model.add(LSTM(100 , activation='relu', input_shape=[length,n_features]))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator, epochs=30)
test_predictions = []
periods = 65
#first_eval_batch = scaled_train[-length:]
first_eval_batch = scaled_full_data[-length:]
current_batch = first_eval_batch.reshape((1,length,n_features))

for i in range(periods):
    current_predict = model.predict(current_batch)[0]
    test_predictions.append(current_predict)
    current_batch = np.append(current_batch[:,1:,:],[[current_predict]] ,axis = 1)
forecast_predictions = full_scaler.inverse_transform(test_predictions)
plt.figure(figsize=(20,10))
plt.title("Chinese data", fontsize=15)
plt.plot(forecast_predictions, label = 'Active prediction')
plt.plot(df_china['active'], label = 'Active real')
plt.xlabel("Ден/Day", fontsize=20)
plt.ylabel("Активни потврдени вкупно /  Active confirmed cases total", fontsize=20)
plt.ylabel("Број на заболени / Infected number", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.legend(fontsize=15)
scaled_full_data_mkd = full_scaler.fit_transform(np.array(df_mkd['active']).reshape(-1,1))
test_predictions = []
periods = 59
first_eval_batch = scaled_full_data_mkd[:length]
#first_eval_batch = scaled_full_data_mkd[-length:]
current_batch = first_eval_batch.reshape((1,length,n_features))

for i in range(periods):
    current_predict = model.predict(current_batch)[0]
    test_predictions.append(current_predict)
    current_batch = np.append(current_batch[:,1:,:],[[current_predict]] ,axis = 1)
forecast_predictions_mkd = full_scaler.inverse_transform(test_predictions)
active = np.array(df_mkd['active'][-25:])
plt.figure(figsize=(20,10))
plt.title('Macedonian data', fontsize=15)
plt.plot(active, label = 'Real active')
plt.plot(forecast_predictions_mkd, label = 'Active prediction')
plt.xlabel("Ден/Day", fontsize=20)
plt.ylabel("Активни потврдени вкупно /  Active confirmed cases total", fontsize=20)
plt.ylabel("Број на заболени / Infected number", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.legend(fontsize=15)

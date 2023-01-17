import pandas as pd

from pandas import datetime

import matplotlib.pyplot as plt

import numpy as np





def parser(y):

    return datetime.strptime(y,'%H')

    #return datetime.strptime(x,'%H') ,date_parser=parser



LTE = pd.read_csv('../input/lte-prediction-cell-2/train_2_cell.csv', parse_dates=[ ['Date','Hour']] )

LTE_test = pd.read_csv('../input/lte-prediction-cell-2/test_2_cell.csv', parse_dates=[ ['Date','Hour']] )

LTE



LTE.dtypes
LTE['Date_Hour'] = LTE['Date_Hour'] + ':00:00'

LTE['Date_Hour'] = pd.to_datetime(LTE['Date_Hour'])

LTE 
LTE.dtypes
LTE_test['Date_Hour'] = LTE_test['Date_Hour'] + ':00:00'

LTE_test['Date_Hour'] = pd.to_datetime(LTE_test['Date_Hour'])

LTE_test
LTE_test.dtypes
LTE_test


LTE = LTE.drop(columns = ['CellName'])

LTE = LTE.set_index('Date_Hour')

LTE = LTE.sort_index()
LTE
LTE_test  = LTE_test.sort_index()

LTE_test = LTE_test.drop(columns = ['CellName'])

LTE_test = LTE_test.set_index('Date_Hour')
LTE_test

LTE.values



LTE.Traffic
## LSTM


from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(LTE)
training_set_scaled
# Creating a data structure with 24 timesteps and 1 output

X_train = []

y_train = []

for i in range(24, 8734):

    X_train.append(training_set_scaled[i-24:i, 0])

    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#pip install --upgrade tensorflow
#pip install keras

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
regressor.fit(X_train, y_train, epochs = 200, batch_size = 50)



LTE_test.size
real_traffic = LTE_test.values
dataset_total = pd.concat((LTE['Traffic'], LTE_test['Traffic']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(LTE_test) - 24:].values

inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)

X_test = []

for i in range(24, 191):

    X_test.append(inputs[i-24:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_traffic = regressor.predict(X_test)

predicted_traffic = sc.inverse_transform(predicted_traffic)

predicted_traffic
plt.plot(real_traffic, color = 'red', label = 'Real LTE Traffic ')

plt.plot(predicted_traffic, color = 'blue', label = 'Predicted LTE Traffic')

plt.title('LSTM Based LTE Traffic Predictor')

plt.xlabel('Input from Test CSV')

plt.ylabel('Traffic')

plt.legend()

plt.show()
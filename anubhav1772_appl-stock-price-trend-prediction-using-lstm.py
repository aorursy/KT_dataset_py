import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import math
from sklearn.metrics import mean_squared_error
train_data = pd.read_csv("../input/train-data/train.csv", header=0)
train_data.head()
test_data = pd.read_csv("../input/test-data/test.csv")
test_data.head()
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data_train = pd.read_csv('../input/train-data/train.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse)
ts = data_train['Open'] 
plt.xlabel('Dates')
plt.ylabel('Open Prices')
plt.plot(ts)
data_test = pd.read_csv('../input/test-data/test.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse)
ts = data_test['Open'] 
plt.xlabel('Dates')
plt.ylabel('Open Prices')
plt.plot(ts)
train = train_data.iloc[:, 1:2].values
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
X_train = []
y_train = []
for i in range(60, train.shape[0]):
    X_train.append(train_scaled[i-60:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
model = Sequential()

# Adding the first LSTM layer 
# Here return_sequences=True means whether to return the last output in the output sequence, or the full sequence.
# it basically tells us that there is another(or more) LSTM layer ahead in the network.
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# Dropout regularisation for tackling overfitting
model.add(Dropout(0.20))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.25))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.25))

model.add(LSTM(units = 50))
model.add(Dropout(0.25))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
# RMSprop is a recommended optimizer as per keras documentation
# check out https://keras.io/optimizers/ for more details
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)
# this will be used later while comparing and visualization
real_stock_price = test_data.iloc[:,1:2].values
# combine original train and test data vertically
# as previous Open Prices are not present in test dataset
# e.g. for predicting Open price for first date in test data, we will need stock open prices on 60 previous dates  
combine = pd.concat((train_data['Open'], test_data['Open']), axis = 0)
# our test inputs also contains stock open Prices of last 60 dates (as described above)
test_inputs = combine[len(combine) - len(test_data) - 60:].values
test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)
test_data.shape
# same steps as we followed while processing training data
X_test = []
for i in range(60, 480):
    X_test.append(test_inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
# inverse_transform because prediction is done on scaled inputs
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
plt.plot(real_stock_price, color = 'red', label = 'Real APPL Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted APPL Stock Price')
plt.title('APPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('APPL Stock Price')
plt.legend()
plt.show()
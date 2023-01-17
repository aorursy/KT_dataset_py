# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load the Data set
testdata = pd.read_csv("/kaggle/input/predice-el-futuro/test_csv.csv")
traindata = pd.read_csv("/kaggle/input/predice-el-futuro/train_csv.csv")
# Only taking the feature coulmn
training_set = traindata.iloc[:, 2:3].values
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with  timesteps and 1 output

X_train = []
y_train = []
for i in range(5, 80):
    X_train.append(training_set_scaled[i-5:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping
# we have to reshape the data into 3D
# 1s is no of lines in X_train ,2nd is no of times step(coulmn od xtrain),3rd is no of predector the features
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
# Initialising the RNN
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
# We use return_sequence true because we are addind anather layer after it, at the last layer it will be False

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
# Adding the output layer
# For Full connection layer we use dense
# As the output is 1D so we use unit=1
regressor.add(Dense(units = 1))
# Compiling the RNN
# For optimizer we can go through keras optimizers Docomentation
# As it is regression problem so we use mean squared error
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
# For best fit accourding to data we can increase the epochs
# For forward & back propageted and update weights we use 5  inputs to train 
regressor.fit(X_train, y_train, epochs = 150, batch_size = 50)
# Getting the real features from the data
dataset_test = pd.read_csv("/kaggle/input/predice-el-futuro/train_csv.csv")
real_feature = dataset_test.iloc[:, 2:3].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((traindata['feature'], dataset_test['feature']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 5:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(5, 80):
    X_test.append(inputs[i-5:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_feature = regressor.predict(X_test)
predicted_feature = sc.inverse_transform(predicted_feature)

# Visualising the results
plt.plot(real_feature, color = 'red', label = 'Real features')
plt.plot(predicted_feature, color = 'blue', label = 'Predicted features')
plt.title('Feature Predection')
plt.xlabel('Time')
plt.ylabel('feature')
plt.legend()
plt.show()

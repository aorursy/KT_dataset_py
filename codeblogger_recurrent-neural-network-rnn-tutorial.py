import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataset = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")[0:5000]

dataset.head()
dataset.info()
dataset.describe().T
train = dataset.loc[:,["AveragePrice"]].values

train
# Feature Scaling

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range = (0, 1))

train_scaled = scaler.fit_transform(train)

train_scaled
plt.figure(figsize=(15, 5));

plt.plot(train_scaled);

plt.show()
dataset.shape[0]
# Creating a data structure with 250 timesteps and 1 output

X_train = []

y_train = []

timesteps = 250



for i in range(timesteps,dataset.shape[0]):

    X_train.append(train_scaled[i-timesteps:i,0])

    y_train.append(train_scaled[i,0])

    

print("X_train length: ", len(X_train))

print("y_train length: ", len(y_train))
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

len(X_train[0])
y_train[0:10]
X_train.shape
# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import SimpleRNN

from keras.layers import Dropout



regressor = Sequential() # initialising the RNN



# adding the first RNN layer and some Droupout regularisation

regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1],1)))

# (X_train.shape[1],1) => output: (250,1)

regressor.add(Dropout(0.2))



# adding the second RNN layer and some Droupout regularisation

regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))

regressor.add(Dropout(0.2))



# adding the third RNN layer and some Droupout regularisation

regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))

regressor.add(Dropout(0.2))



# adding a fourth RNN layer and some Dropout regularisation

regressor.add(SimpleRNN(units=50))

regressor.add(Dropout(0.2))



# adding the output layer

regressor.add(Dense(units = 1))



# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')



# Fitting the RNN to the Training set

regressor.fit(X_train, y_train, epochs = 20, batch_size = 132)
dataset_test = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")[5000:10000]

dataset_test.head()
test = dataset_test.loc[:,["AveragePrice"]].values

test
# Getting the predicted stock price of 2017

dataset_total = pd.concat((dataset['AveragePrice'], dataset_test['AveragePrice']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - timesteps:].values.reshape(-1,1)

inputs = scaler.transform(inputs)  # min max scaler

inputs
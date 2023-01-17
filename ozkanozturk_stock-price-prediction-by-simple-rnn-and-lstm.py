# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/tesla-stock-price/Tesla.csv - Tesla.csv.csv")
data.head()
data.info()
length_data = len(data)     # rows that data has

split_ratio = 0.7           # %70 train + %30 validation

length_train = round(length_data * split_ratio)  

length_validation = length_data - length_train

print("Data length :", length_data)

print("Train data length :", length_train)

print("Validation data lenth :", length_validation)
train_data = data[:length_train].iloc[:,:2] 

train_data['Date'] = pd.to_datetime(train_data['Date'])  # converting to date time object

train_data
validation_data = data[length_train:].iloc[:,:2]

validation_data['Date'] = pd.to_datetime(validation_data['Date'])  # converting to date time object

validation_data
dataset_train = train_data.Open.values

dataset_train.shape
# Change 1d array to 2d array

# Changing shape from (1692,) to (1692,1)

dataset_train = np.reshape(dataset_train, (-1,1))

dataset_train.shape
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))





# scaling dataset

dataset_train_scaled = scaler.fit_transform(dataset_train)



dataset_train_scaled.shape
plt.subplots(figsize = (15,6))

plt.plot(dataset_train_scaled)

plt.xlabel("Days as 1st, 2nd, 3rd..")

plt.ylabel("Open Price")

plt.show()
X_train = []

y_train = []



time_step = 50



for i in range(time_step, length_train):

    X_train.append(dataset_train_scaled[i-time_step:i,0])

    y_train.append(dataset_train_scaled[i,0])

    

# convert list to array

X_train, y_train = np.array(X_train), np.array(y_train)

print("Shape of X_train before reshape :",X_train.shape)

print("Shape of y_train before reshape :",y_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

y_train = np.reshape(y_train, (y_train.shape[0],1))



print("Shape of X_train after reshape :",X_train.shape)

print("Shape of y_train after reshape :",y_train.shape)
X_train[0]
y_train[0]
# importing libraries

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import SimpleRNN

from keras.layers import Dropout



# initializing the RNN

regressor = Sequential()



# adding first RNN layer and dropout regulatization

regressor.add(

    SimpleRNN(units = 50, 

              activation = "tanh", 

              return_sequences = True, 

              input_shape = (X_train.shape[1],1))

             )



regressor.add(

    Dropout(0.2)

             )





# adding second RNN layer and dropout regulatization



regressor.add(

    SimpleRNN(units = 50, 

              activation = "tanh", 

              return_sequences = True)

             )



regressor.add(

    Dropout(0.2)

             )



# adding third RNN layer and dropout regulatization



regressor.add(

    SimpleRNN(units = 50, 

              activation = "tanh", 

              return_sequences = True)

             )



regressor.add(

    Dropout(0.2)

             )



# adding fourth RNN layer and dropout regulatization



regressor.add(

    SimpleRNN(units = 50)

             )



regressor.add(

    Dropout(0.2)

             )



# adding the output layer

regressor.add(Dense(units = 1))



# compiling RNN

regressor.compile(

    optimizer = "adam", 

    loss = "mean_squared_error",

    metrics = ["accuracy"])



# fitting the RNN

history = regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)
# Losses

history.history["loss"]
# Plotting Loss vs Epochs

plt.figure(figsize =(10,7))

plt.plot(history.history["loss"])

plt.xlabel("Epochs")

plt.ylabel("Losses")

plt.title("Simple RNN model, Loss vs Epoch")

plt.show()
# Plotting Accuracy vs Epochs

plt.figure(figsize =(10,5))

plt.plot(history.history["accuracy"])

plt.xlabel("Epochs")

plt.ylabel("Accuracies")

plt.title("Simple RNN model, Accuracy vs Epoch")

plt.show()
y_pred = regressor.predict(X_train)  # predictions

y_pred = scaler.inverse_transform(y_pred) # scaling back from 0-1 to original

y_pred.shape
y_train = scaler.inverse_transform(y_train) # scaling back from 0-1 to original

y_train.shape
# visualisation

plt.figure(figsize = (30,10))

plt.plot(y_pred, color = "b", label = "y_pred" )

plt.plot(y_train, color = "g", label = "y_train")

plt.xlabel("Days")

plt.ylabel("Open price")

plt.title("Simple RNN model, Predictions with input X_train vs y_train")

plt.legend()

plt.show()

dataset_validation = validation_data.Open.values  # getting "open" column and converting to array

dataset_validation = np.reshape(dataset_validation, (-1,1))  # converting 1D to 2D array

scaled_dataset_validation =  scaler.fit_transform(dataset_validation)  # scaling open values to between 0 and 1

print("Shape of scaled validation dataset :",scaled_dataset_validation.shape)
# Creating X_test and y_test

X_test = []

y_test = []



for i in range(time_step, length_validation):

    X_test.append(scaled_dataset_validation[i-time_step:i,0])

    y_test.append(scaled_dataset_validation[i,0])
# Converting to array

X_test, y_test = np.array(X_test), np.array(y_test)
print("Shape of X_test before reshape :",X_test.shape)

print("Shape of y_test before reshape :",y_test.shape)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))  # reshape to 3D array

y_test = np.reshape(y_test, (-1,1))  # reshape to 2D array
print("Shape of X_test after reshape :",X_test.shape)

print("Shape of y_test after reshape :",y_test.shape)
# predictions with X_test data

y_pred_of_test = regressor.predict(X_test)

# scaling back from 0-1 to original

y_pred_of_test = scaler.inverse_transform(y_pred_of_test) 

print("Shape of y_pred_of_test :",y_pred_of_test.shape)
# visualisation

plt.figure(figsize = (30,10))

plt.plot(y_pred_of_test, label = "y_pred_of_test", c = "orange")

plt.plot(scaler.inverse_transform(y_test), label = "y_test", c = "g")

plt.xlabel("Days")

plt.ylabel("Open price")

plt.title("Simple RNN model, Prediction with input X_test vs y_test")

plt.legend()

plt.show()
# Visualisation

plt.subplots(figsize =(30,12))

plt.plot(train_data.Date, train_data.Open, label = "train_data", color = "b")

plt.plot(validation_data.Date, validation_data.Open, label = "validation_data", color = "g")

plt.plot(train_data.Date.iloc[time_step:], y_pred, label = "y_pred", color = "r")

plt.plot(validation_data.Date.iloc[time_step:], y_pred_of_test, label = "y_pred_of_test", color = "orange")

plt.xlabel("Days")

plt.ylabel("Open price")

plt.title("Simple RNN model, Train-Validation-Prediction")

plt.legend()

plt.show()
y_train = scaler.fit_transform(y_train)
from keras.layers import LSTM



model_lstm = Sequential()

model_lstm.add(

    LSTM(64,return_sequences=True,input_shape = (X_train.shape[1],1))) #64 lstm neuron block

model_lstm.add(

    LSTM(64, return_sequences= False))

model_lstm.add(Dense(32))

model_lstm.add(Dense(1))

model_lstm.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])

history2 = model_lstm.fit(X_train, y_train, epochs = 10, batch_size = 10)

plt.figure(figsize =(10,5))

plt.plot(history2.history["loss"])

plt.xlabel("Epochs")

plt.ylabel("Losses")

plt.title("LSTM model, Accuracy vs Epoch")

plt.show()
plt.subplots(figsize =(30,12))

plt.plot(scaler.inverse_transform(model_lstm.predict(X_test)), label = "y_pred_of_test", c = "orange" )

plt.plot(scaler.inverse_transform(y_test), label = "y_test", color = "g")

plt.xlabel("Days")

plt.ylabel("Open price")

plt.title("LSTM model, Predictions with input X_test vs y_test")

plt.legend()

plt.show()
data.iloc[-1]
X_input = data.iloc[-time_step:].Open.values               # getting last 50 rows and converting to array

X_input = scaler.fit_transform(X_input.reshape(-1,1))      # converting to 2D array and scaling

X_input = np.reshape(X_input, (1,50,1))                    # reshaping : converting to 3D array

print("Shape of X_input :", X_input.shape)

X_input
simple_RNN_prediction = scaler.inverse_transform(regressor.predict(X_input))

LSTM_prediction = scaler.inverse_transform(model_lstm.predict(X_input))

print("Simple RNN, Open price prediction for 3/18/2017      :", simple_RNN_prediction[0,0])

print("LSTM prediction, Open price prediction for 3/18/2017 :", LSTM_prediction[0,0])
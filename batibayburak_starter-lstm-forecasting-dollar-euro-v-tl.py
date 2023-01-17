import warnings

import itertools

from math import sqrt

from datetime import datetime

from numpy import concatenate

import numpy as np

import pandas as pd

import math

 

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import LSTM, Bidirectional, GRU

from keras.layers.recurrent import LSTM

from sklearn.utils import shuffle

 

import plotly.offline as py

import plotly.graph_objs as go

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

 

py.init_notebook_mode(connected=True)

plt.style.use('fivethirtyeight')

import os

os.listdir("../input/dolar-euro")
df = pd.read_csv("../input/dolar-euro/EVDS.csv", sep = ";", parse_dates = ["Tarih"])

df.head()
df.info
df.describe()
# Indexing and filling null values with interpolation for resample data values

df.index = df.Tarih

df['Dolar'].interpolate(method='linear', inplace=True)

df['Euro'].interpolate(method='linear', inplace=True)

df["Dolar"] = df["Dolar"].str.replace(",", ".").astype("float32")

df["Euro"] = df["Euro"].str.replace(",", ".").astype("float32")
# Control non-null values

df.info()
# Figurize orijinal data values

fig = plt.figure(figsize=(20,8))

df.Dolar.plot(label='Dolar')

df.Euro.plot(label='Euro')

plt.legend(loc='best')

plt.title('Daily Exchange Buy Rates', fontsize=14)

plt.show()
# Normalization and scaler values with some example for Dolar column

values = df['Dolar'].values.reshape(-1,1)

values = values.astype('float64')

scaler = MinMaxScaler(feature_range=(0, 1))

dataset1 = scaler.fit_transform(values)

dataset1[0:5]
# Normalization and scaler values with some example for Euro column

values = df['Euro'].values.reshape(-1,1)

values = values.astype('float64')

scaler = MinMaxScaler(feature_range=(0, 1))

dataset2 = scaler.fit_transform(values)

dataset2[0:5]
# Build Model

TRAIN_SIZE = 0.80

 

train_size1 = int(len(dataset1) * TRAIN_SIZE)

test_size1 = len(dataset1) - train_size1

train1, test1 = dataset1[0:train_size1, :], dataset1[train_size1:len(dataset1), :]

print("Number of Days (training set, test set): " + str((len(train1), len(test1))))
# For Dolar Colums, create using dataset with train set

def create_dataset1(dataset1, window_size = 1):

    data_X1, data_Y1 = [], []

    for i in range(len(dataset1) - window_size - 1):

        a = dataset1[i:(i + window_size), 0]

        data_X1.append(a)

        data_Y1.append(dataset1[i + window_size, 0])

    return(np.array(data_X1), np.array(data_Y1))
# Dolar data values are shaped and training

window_size = 1

train_X1, train_Y1 = create_dataset1(train1, window_size)

test_X1, test_Y1 = create_dataset1(test1, window_size)

print("Original training data shape:")

print(train_X1.shape)

 



train_X1 = np.reshape(train_X1, (train_X1.shape[0], 1, train_X1.shape[1]))

test_X1 = np.reshape(test_X1, (test_X1.shape[0], 1, test_X1.shape[1]))

print("New training data shape:")

print(train_X1.shape)
# Fit the model for Dolar data values

def fit_model1(train_X1, train_Y1, window_size = 1):

    model = Sequential()

    model.add(LSTM(100, 

                   input_shape = (1, window_size)))

    model.add(Dense(1))

    model.compile(loss = "mean_squared_error", 

                  optimizer = "adam")

    model.fit(train_X1, 

              train_Y1, 

              epochs = 30, 

              batch_size = 1, 

              verbose = 1)

    

    return(model)

 



model1 = fit_model1(train_X1, train_Y1, window_size)





# Find the prediction value with Dolar data values

def predict_and_score1(model1, X1, Y1):

   

    pred = scaler.inverse_transform(model1.predict(X1))

    orig_data = scaler.inverse_transform([Y1])

    

    score1 = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))

    return(score1, pred)

 

rmse_train1, train_predict1 = predict_and_score1(model1, train_X1, train_Y1)

rmse_test1, test_predict1 = predict_and_score1(model1, test_X, test_Y1)

 

print("Training data score: %.2f RMSE" % rmse_train1)

print("Test data score: %.2f RMSE" % rmse_test1)



#The graphic drawn with prediction and orijinal data values for Dolar column

train_predict_plot1 = np.empty_like(dataset1)

train_predict_plot1[:, :] = np.nan

train_predict_plot1[window_size:len(train_predict1) + window_size, :] = train_predict1

 



test_predict_plot1 = np.empty_like(dataset1)

test_predict_plot1[:, :] = np.nan

test_predict_plot1[len(train_predict1) + (window_size * 2) + 1:len(dataset1) - 1, :] = test_predict1



plt.figure(figsize = (15, 5))

plt.plot(scaler.inverse_transform(dataset1), label = "True value")

plt.plot(train_predict_plot1, label = "Training set prediction")

plt.plot(test_predict_plot1, label = "Test set prediction")

plt.xlabel("Days")

plt.ylabel("Exchange Rates")

plt.title("Comparison true vs. predicted training / test Dolar")

plt.legend()

plt.show()



# The same way for Euro column with Euro data values, build some model

train_size2 = int(len(dataset2) * TRAIN_SIZE)

test_size2 = len(dataset2) - train_size2

train2, test2 = dataset2[0:train_size2, :], dataset1[train_size2:len(dataset2), :]

print("Number of Days (training set, test set): " + str((len(train2), len(test2))))



# For Dolar Colums, create using dataset with train set

def create_dataset2(dataset2, window_size = 1):

    data_X2, data_Y2 = [], []

    for i in range(len(dataset2) - window_size - 1):

        a = dataset2[i:(i + window_size), 0]

        data_X2.append(a)

        data_Y2.append(dataset2[i + window_size, 0])

    return(np.array(data_X2), np.array(data_Y2))



# Euro data values are shaped and training

window_size = 1

train_X2, train_Y2 = create_dataset2(train2, window_size)

test_X2, test_Y2 = create_dataset2(test2, window_size)

print("Original training data shape:")

print(train_X2.shape)

 



train_X2 = np.reshape(train_X2, (train_X2.shape[0], 1, train_X2.shape[1]))

test_X2 = np.reshape(test_X2, (test_X2.shape[0], 1, test_X2.shape[1]))

print("New training data shape:")

print(train_X2.shape)



# Fit the model for Euro data values



def fit_model2(train_X2, train_Y2, window_size = 1):

    model = Sequential()

   

    model.add(LSTM(100, 

                   input_shape = (1, window_size)))

    model.add(Dense(1))

    model.compile(loss = "mean_squared_error", 

                  optimizer = "adam")

  

    model.fit(train_X2, 

              train_Y2, 

              epochs = 30, 

              batch_size = 1, 

              verbose = 1)

    

    return(model)

model2 = fit_model(train_X2, train_Y2, window_size)





# Find the prediction value with Euro data values

def predict_and_score2(model2, X2, Y2):

   

    pred = scaler.inverse_transform(model.predict(X2))

    orig_data = scaler.inverse_transform([Y2])

    

    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))

    return(score, pred)

 

rmse_train2, train_predict2 = predict_and_score(model2, train_X2, train_Y2)

rmse_test2, test_predict2 = predict_and_score(model2, test_X2, test_Y2)

 

print("Training data score: %.2f RMSE" % rmse_train2)

print("Test data score: %.2f RMSE" % rmse_test2)



 

#The graphic drawn with prediction and orijinal data values for Euro column    

    

train_predict_plot2 = np.empty_like(dataset2)

train_predict_plot2[:, :] = np.nan

train_predict_plot2[window_size:len(train_predict2) + window_size, :] = train_predict2

 



test_predict_plot2 = np.empty_like(dataset2)

test_predict_plot2[:, :] = np.nan

test_predict_plot2[len(train_predict2) + (window_size * 2) + 1:len(dataset2) - 1, :] = test_predict2



plt.figure(figsize = (15, 5))

plt.plot(scaler.inverse_transform(dataset2), label = "True value")

plt.plot(train_predict_plot2, label = "Training set prediction")

plt.plot(test_predict_plot2, label = "Test set prediction")

plt.xlabel("Days")

plt.ylabel("Exchange Rates")

plt.title("Comparison true vs. predicted training / test  Euro")

plt.legend()

plt.show()







#The graphic drawn with prediction and orijinal data values for Euro and Dolar column together



train_predict_plot1 = np.empty_like(dataset1)

train_predict_plot1[:, :] = np.nan

train_predict_plot1[window_size:len(train_predict1) + window_size, :] = train_predict1

train_predict_plot2 = np.empty_like(dataset2)

train_predict_plot2[:, :] = np.nan

train_predict_plot2[window_size:len(train_predict2) + window_size, :] = train_predict2



test_predict_plot1 = np.empty_like(dataset1)

test_predict_plot1[:, :] = np.nan

test_predict_plot1[len(train_predict1) + (window_size * 2) + 1:len(dataset1) - 1, :] = test_predict1

test_predict_plot2 = np.empty_like(dataset2)

test_predict_plot2[:, :] = np.nan

test_predict_plot2[len(train_predict2) + (window_size * 2) + 1:len(dataset2) - 1, :] = test_predict2



plt.figure(figsize = (15, 5))

plt.plot(scaler.inverse_transform(dataset1), label = "True value")

plt.plot(train_predict_plot1, label = "Dolar Training set prediction")

plt.plot(test_predict_plot1, label = "Dolar Test set prediction")

plt.plot(train_predict_plot2, label = "Euro Training set prediction")

plt.plot(test_predict_plot2, label = "Euro Test set prediction")

plt.xlabel("Times")

plt.ylabel("Exchange Rates")

plt.title("Comparison true vs. predicted training / test Dolar and Euro")

plt.legend()

plt.show()
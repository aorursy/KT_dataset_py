# RIO Tinto --> one of the largest mining corporations in the world
# library imports
#import os
#import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from tensorflow import random

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
tf.__version__
data = pd.read_csv("../input/random-stock-from-yahoo-finance/RIO.csv", index_col= 0, parse_dates= True)
data.shape
data.head()
data.tail()
training = data[["Adj Close"]].values
"""
scaling the data --> 

"""
scaler = MinMaxScaler(feature_range = (0, 1))

training_scaled= scaler.fit_transform(training)

testing_scaled = training_scaled[-400-60:]
training_scaled = training_scaled[-1600:-400]
print(len(testing_scaled), len(training_scaled))
def prepare_train_test(training_scaled, testing_scaled):
    x_train = []
    y_train = []
    for i in range(60, len(training_scaled)):
        x_train.append(training_scaled[i-60:i, 0])
        y_train.append(training_scaled[i, 0])
    
    x_test= []
    y_test = []
    
    for i in range(60, len(testing_scaled)):
        x_test.append(testing_scaled[i-60:i, 0])
        y_test.append(testing_scaled[i, 0])
    
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_train, y_train, x_test, y_test
X_train, y_train, X_test, y_test= prepare_train_test(training_scaled, testing_scaled)
print(X_test.shape, y_test.shape, X_train.shape, y_train.shape)
def get_model():
    
    model= Sequential()
    model.add(LSTM(units = 200, return_sequences  = True, input_shape = (X_train.shape[1], 1)))
    model.add(LSTM(units = 200, return_sequences = True))
    
    model.add(LSTM(units = 100, return_sequences = True))
    model.add(LSTM(units = 100))
    model.add(Dense(units = 1))
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return model
model = get_model()
model.summary()
history = model.fit(X_train, y_train, epochs = 100, batch_size = 60)

#!pip install tensorflow-gpu
def training_loss_graph(history):
    plt.plot(history.history['loss'], label = 'Training  Loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.show()
training_loss_graph(history)
def get_predicted_INV_scaled(X_test):
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    
    prices = scaler.inverse_transform([y_test])
    return prices, predicted_prices
prices, predicted_prices = get_predicted_INV_scaled(X_test)
def show_graph_result(prices, predicted_prices):
    index = data.index.values[-len(prices[0]):]
    test_result = pd.DataFrame(columns = ['real', 'predicted'])
    test_result['real'] = prices[0]
    test_result['predicted'] = predicted_prices
    test_result.index = index
    
    test_result.plot(figsize = (16, 10))
    plt.title("Actual and Predicted")
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.show()
show_graph_result(prices, predicted_prices)
# next 
# methods for you to build what if your data for the future --> 
# 
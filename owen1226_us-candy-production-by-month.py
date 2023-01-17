# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline

import re
import math

from keras.utils import to_categorical, layer_utils, plot_model
# Use 3 years data in time steps 
NUM_TIME_STEPS = 36
df = pd.read_csv("../input/candy_production.csv")
df_mean = df['IPG3113N'].mean()
df_max = df['IPG3113N'].max()
df_min = df['IPG3113N'].min()
df_norm = (df['IPG3113N'] - df_mean) / (df_max - df_min)
df.head(10)
from sklearn.model_selection import train_test_split

def preprocess_dataset(df, test_size):
    m = len(df) - NUM_TIME_STEPS - 1
    X = np.zeros((m, NUM_TIME_STEPS, 12))
    Y = np.zeros((m))
    
    for i in range(m):
        for j in range(NUM_TIME_STEPS):
            X[i, j, j%12] = df.iloc[i+j]
        Y[i] = df.iloc[i+NUM_TIME_STEPS+1]
    
#     return train_test_split(X, Y, test_size = test_size, shuffle = False, stratify = None)
    return train_test_split(X, Y, test_size = test_size, random_state = 1)
    
    
    
X_train, X_test, Y_train, Y_test = preprocess_dataset(df_norm, test_size=0.02)
print("X_train", X_train.shape)
print("Y_train", Y_train.shape)
print("X_test", X_test.shape)
print("Y_test", Y_test.shape)
# print(X_train[0])
from keras.layers import Embedding, Input, Dropout, SpatialDropout1D, LSTM, Dense, Activation
from keras.initializers import glorot_uniform, he_uniform
from keras.models import Model, Sequential

def MyModel(input_shape):
    X_input = Input(input_shape, dtype="float32")

    # Add dropout with a probability
    X = SpatialDropout1D(0.3)(X_input)
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(X)
    
    # Add dropout with a probability
#     X = Dropout(0.5)(X)
    
    # Propagate X trough another LSTM layer with 32-dimensional hidden state, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(X)
    
    # Add dropout with a probability
#     X = Dropout(0.5)(X)
    
    # Propagate X through a Dense layer without activation to get back a batch of 1-dimensional vectors.
    X = Dense(1)(X)
    
    # Create Model instance.
    model = Model(inputs = X_input, outputs = X)

    return model
model = MyModel((NUM_TIME_STEPS, 12))
model.summary()
from keras.optimizers import Adam

model.compile(loss='mse', optimizer=Adam(lr=0.00003, decay=1e-6, beta_1=0.9, beta_2=0.999))
from keras.callbacks import EarlyStopping, ModelCheckpoint

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', verbose=1, save_best_only=True)   # Save the best model
hist = model.fit(X_train, Y_train, batch_size = 32, epochs = 100, verbose=1, callbacks=[monitor, checkpoint], validation_split=0.01, shuffle=True)
model.load_weights('best_weights.hdf5')
model.save('final_model.h5')
def plot_train_history(history):
    # plot the mse cost 
    loss_list = history['loss']
    val_loss_list = history['val_loss']

    # plot the cost
    plt.plot(loss_list, 'b', label='Training cost')
    plt.plot(val_loss_list, 'r', label='Validation cost')
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('Training and validation cost')
    plt.legend()

    

plot_train_history(hist.history)
score = model.evaluate(X_test, Y_test)
print ("Test Loss = " + str(score))
Y_test_pred = model.predict(X_test, batch_size=32, verbose=1).squeeze()
from sklearn.metrics import mean_squared_error

def plot_predictive_interval(Y_test, Y_test_pred, z):
    rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
    interval = z * rmse
    
    plt.plot(Y_test_pred, 'b', label='labels')
    plt.plot(Y_test, 'r', label='predictions')
    plt.plot(Y_test + interval, 'g')
    plt.plot(Y_test - interval, 'g')
    plt.ylabel('amounts')
    plt.xlabel('time steps')
    plt.title('labels & predictions')
    plt.legend()

    
    
Y_test2 = Y_test * (df_max - df_min) + df_mean
Y_test_pred2 = Y_test_pred * (df_max - df_min) + df_mean
# print(Y_test2)
# print(Y_test_pred2)

# z = 1.96, by using 95% predictive interval
plot_predictive_interval(Y_test2, Y_test_pred2, 1.96)

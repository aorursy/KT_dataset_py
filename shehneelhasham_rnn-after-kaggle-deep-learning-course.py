from numpy import array
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
data = data.reshape((1, 10, 1))
print(data.shape)
import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Flatten

img_rows, img_cols = 28, 28 
num_classes = 10

def prep_data(raw): 
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:] #gives the data as a numpy array
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data)
print("Data Loaded")
print(fashion_data.shape)
print(x.shape)
print("x.shape[1:]", x.shape[1:])
model_LSTM = Sequential()

model_LSTM.add(LSTM(units = 128, activation = 'relu', input_shape = (x.shape[1:]), return_sequences = False)) 
#units = 1 for a vanilla RNN as it is only one layer of the vectors described above
#can have return_sequences=True in case we want to continue onto another LSTM, but False is the default
model_LSTM.add(Dropout(0.2))
#we are adding dropout to ignore randomly chosen nodes, this reduces overfitting
#like in the Deep Learning course

model_LSTM.add(Dense(12, activation = 'relu'))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(Dense(num_classes, activation = 'softmax')) #prediction layer

model_LSTM.compile(loss = keras.losses.categorical_crossentropy,
                   optimizer = 'adam',
                   metrics = ['accuracy'])

model_LSTM.summary()
model_LSTM.fit(x,y, batch_size = 100, epochs = 4, validation_split = 0.2) #validation_split = 0.2 means that we set 
#20% of the data aside for validation
model_GRU = Sequential()

model_GRU.add(GRU(128, activation = 'relu', input_shape = (x.shape[1:]), return_sequences = False)) 
model_GRU.add(Dropout(0.2))

model_GRU.add(Flatten())

model_GRU.add(Dense(12, activation = 'relu'))
model_GRU.add(Dropout(0.2))

model_GRU.add(Dense(num_classes, activation = 'softmax')) #prediction layer

opt = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-5)

model_GRU.compile(loss = keras.losses.categorical_crossentropy,
                   optimizer = opt,
                   metrics = ['accuracy'])

model_GRU.summary()
model_GRU.fit(x,y,batch_size = 100, epochs = 4, validation_split = 0.2)
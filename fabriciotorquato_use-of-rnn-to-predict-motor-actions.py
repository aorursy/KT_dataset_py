import matplotlib.pyplot as plt

import numpy as np 

import os

import pandas as pd
from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import LSTM, Dense, BatchNormalization
from keras.utils import Sequence

class SeqGen(Sequence):



    def __init__(self, x_set, y_set, batch_size):

        self.x, self.y = x_set, y_set

        self.batch_size = batch_size



    def __len__(self):

        return int(np.ceil(len(self.x) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]



        return batch_x, batch_y
def get_model():

    model = Sequential()

    model.add(LSTM(16,input_shape=(11,11), return_sequences=True))    

    model.add(LSTM(16))  

    model.add(BatchNormalization())  

    model.add(Dense(3, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
def plot_history_accuracy(history):

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
def plot_history_loss(history):

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
def train_model(model, X, Y):

    x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=42, shuffle=True )

    standard = StandardScaler().fit(x_train)

    x_train_standard = standard.transform(x_train).reshape(2016,11,11)

    x_test_standard = standard.transform(x_test).reshape(864,11,11)

    return model.fit_generator(SeqGen(x_train_standard,y_train,batch_size=12), validation_data=(x_test_standard,y_test), epochs=20, verbose=1)
df = pd.read_csv('/kaggle/input/eeg-data-from-hands-movement/Dataset/user_a.csv', delimiter=',', index_col=False)

df.dataframeName = 'dataset.csv'
X = df.iloc[:,1:]

Y = df.iloc[:,0]

l = ['complement'] * (121 - X.shape[1]) 



for index,col in enumerate(l):

    X[col+str(index)] = 0



X = X.values

Y = Y.values
encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)

dummy_y = np_utils.to_categorical(encoded_Y)
model = get_model()

history = train_model(model, X, dummy_y)
plot_history_accuracy(history)
plot_history_loss(history)
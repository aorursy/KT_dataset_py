import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import os

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))



# print(tf.VERSION)

print(tf.keras.__version__)



rock_dataset = pd.read_csv("../input/emg-4/0.csv", header=None) # class = 0

scissors_dataset = pd.read_csv("../input/emg-4/1.csv", header=None) # class = 1

paper_dataset = pd.read_csv("../input/emg-4/2.csv", header=None) # class = 2

ok_dataset = pd.read_csv("../input/emg-4/3.csv", header=None) # class = 3



frames = [rock_dataset, scissors_dataset, paper_dataset, ok_dataset]

dataset = pd.concat(frames)



dataset_train = dataset.iloc[np.random.permutation(len(dataset))]

dataset_train.reset_index(drop=True)



X_train = []

y_train = []



for i in range(0, dataset_train.shape[0]):

    row = np.array(dataset_train.iloc[i:1+i, 0:64].values)

    X_train.append(np.reshape(row, (64, 1)))

    y_train.append(np.array(dataset_train.iloc[i:1+i, -1:])[0][0])

    

X_train = np.array(X_train)

y_train = np.array(y_train)



# Reshape to one flatten vector

X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1], 1)

X_train = sc.fit_transform(X_train)



# Reshape again after normalization to (-1, 8, 8)

X_train = X_train.reshape((-1, 8, 8))



# Convert to one hot

y_train = np.eye(np.max(y_train) + 1)[y_train]



print("All Data size X and y")

print(X_train.shape)

print(y_train.shape)
# Splitting Train/Test

X_test = X_train[7700:]

y_test = y_train[7700:]

print("Test Data size X and y")

print(X_test.shape)

print(y_test.shape)



X_train = X_train[0:7700]

y_train = y_train[0:7700]

print("Train Data size X and y")

print(X_train.shape)

print(y_train.shape)
X_train.shape
# Creating the model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import *

from tensorflow.keras.layers import Dropout



classifier = Sequential()



classifier.add(GRU(units=32, return_sequences=True, input_shape=(X_train.shape[1], 8)))

classifier.add(Dropout(0.2))



classifier.add(GRU(units = 64, return_sequences = True))

classifier.add(Dropout(0.2))



classifier.add(Dense(units = 100))

classifier.add(BatchNormalization())



classifier.add(GRU(units = 100, return_sequences = True))

classifier.add(Dropout(0.2))



classifier.add(GRU(units = 64))

classifier.add(Dropout(0.2))



classifier.add(Dense(units = 64, activation="relu"))

classifier.add(Dense(units = 128, activation="relu"))



classifier.add(Dense(units = 4, activation="softmax"))



classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy')



classifier.fit(X_train, y_train, epochs = 250, batch_size = 8, verbose=2, callbacks=[keras.callbacks.EarlyStopping(restore_best_weights=True, patience=10),keras.callbacks.ReduceLROnPlateau(patience=7) ])



# Save

classifier.save("model_cross_splited_data.h5")

print("Saved model to disk")
###############################################



from tensorflow import keras



# # Load Model

# model = keras.models.load_model('model_cross_splited_data.h5')

# model.summary()



def evaluateModel(prediction, y):

    good = 0

    for i in range(len(y)):

        if (prediction[i] == np.argmax(y[i])):

            good = good +1

    return (good/len(y)) * 100.0



result_test = classifier.predict_classes(X_test)

print("Correct classification rate on test data")

print(evaluateModel(result_test, y_test))



result_train = classifier.predict_classes(X_train)

print("Correct classification rate on train data")

print(evaluateModel(result_train, y_train))
# Creating the model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import *

from tensorflow.keras.layers import Dropout



classifier = Sequential()



classifier.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 8)))

classifier.add(Dropout(0.2))



classifier.add(LSTM(units = 64, return_sequences = True))

classifier.add(Dropout(0.2))



classifier.add(Dense(units = 100))

classifier.add(BatchNormalization())



classifier.add(LSTM(units = 100, return_sequences = True))

classifier.add(Dropout(0.2))



classifier.add(LSTM(units = 64))

classifier.add(Dropout(0.2))



classifier.add(Dense(units = 64, activation="relu"))

classifier.add(Dense(units = 128, activation="relu"))



classifier.add(Dense(units = 4, activation="softmax"))



classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy')



classifier.fit(X_train, y_train, epochs = 250, batch_size = 8, verbose=2, callbacks=[keras.callbacks.EarlyStopping(restore_best_weights=True, patience=10),keras.callbacks.ReduceLROnPlateau(patience=7) ])



# Save

classifier.save("model_cross_splited_data.h5")

print("Saved model to disk")



###############################################



from tensorflow import keras



# # Load Model

# model = keras.models.load_model('model_cross_splited_data.h5')

# model.summary()



def evaluateModel(prediction, y):

    good = 0

    for i in range(len(y)):

        if (prediction[i] == np.argmax(y[i])):

            good = good +1

    return (good/len(y)) * 100.0



result_test = classifier.predict_classes(X_test)

print("Correct classification rate on test data")

print(evaluateModel(result_test, y_test))



result_train = classifier.predict_classes(X_train)

print("Correct classification rate on train data")

print(evaluateModel(result_train, y_train))
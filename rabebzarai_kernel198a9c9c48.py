from __future__ import print_function

import numpy as np

np.random.seed(1337)  # for reproducibility



from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Lambda

from keras.layers import Embedding

from keras.layers import Convolution1D,MaxPooling1D, Flatten

from keras.datasets import imdb

from keras import backend as K

from sklearn.model_selection import train_test_split

import pandas as pd

from keras.utils.np_utils import to_categorical



from sklearn.preprocessing import Normalizer

from keras.models import Sequential

from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D

from keras.utils import np_utils

import numpy as np

import h5py

from keras import callbacks

from keras.layers import LSTM, GRU, SimpleRNN

from keras.callbacks import CSVLogger

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from sklearn.metrics import (precision_score, confusion_matrix, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)

from sklearn import metrics

train= pd.read_csv('../input/kddcup199/Train.csv')

test = pd.read_csv('../input/kddcup199/Test.csv')
X = train.iloc[:,1:21]

Y = train.iloc[:,0]

C = test.iloc[:,0]

T = test.iloc[:,1:21]

trainX = np.array(X)

testT = np.array(T)

trainX.astype(float)

testT.astype(float)

scaler = Normalizer().fit(trainX)

trainX = scaler.transform(trainX)

scaler = Normalizer().fit(testT)

testT = scaler.transform(testT)

y_train = np.array(Y)

y_test = np.array(C)

X_train = np.array(trainX)

X_test = np.array(testT)
X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))

X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))

from tensorflow.keras import layers

import tensorflow as tf
lstm_output_size = 128



cnn = Sequential()

cnn.add(Convolution1D(64, 3, border_mode="same",activation="relu",input_shape=(20, 1)))

cnn.add(Convolution1D(64, 3, border_mode="same", activation="relu"))

cnn.add(MaxPooling1D(pool_length=(2)))

cnn.add(Convolution1D(128, 3, border_mode="same", activation="relu"))

cnn.add(Convolution1D(128, 3, border_mode="same", activation="relu"))

cnn.add(MaxPooling1D(pool_length=(2)))

cnn.add(Flatten())

cnn.add(Dense(128, activation="relu"))

cnn.add(Dropout(0.5))

cnn.add(Dense(1, activation="sigmoid"))
cnn.summary()
checkpointer = callbacks.ModelCheckpoint(filepath="checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')

csv_logger = CSVLogger('training_set_dnnanalysis.csv',separator=',', append=False)

cnn.fit(X_train, y_train, batch_size=64, nb_epoch=50, callbacks=[checkpointer,csv_logger])

cnn.save("cnnlayer_model.hdf5")

precision_score(train, test)
#traindata = pd.read_csv('kdd/kddtrain.csv', header=None)

test = pd.read_csv('../input/kddcup199/Test.csv')



C = test.iloc[:,0]

T = test.iloc[:,1:21]



scaler = Normalizer().fit(testT)

testT = scaler.transform(testT)



y_test = np.array(C)

X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))





lstm_output_size = 128



cnn = Sequential()

cnn.add(Convolution1D(64, 3, border_mode="same",activation="relu",input_shape=(41, 1)))

cnn.add(Convolution1D(64, 3, border_mode="same", activation="relu"))

cnn.add(MaxPooling1D(pool_length=(2)))

cnn.add(Convolution1D(128, 3, border_mode="same", activation="relu"))

cnn.add(Convolution1D(128, 3, border_mode="same", activation="relu"))

cnn.add(MaxPooling1D(pool_length=(2)))

cnn.add(Flatten())

cnn.add(Dense(128, activation="relu"))

cnn.add(Dropout(0.5))

cnn.add(Dense(1, activation="sigmoid"))

from keras.utils.np_utils import to_categorical

from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)

from sklearn import metrics
testdata = pd.read_csv('predicted.txt', header=None)

traindata = pd.read_csv('expected.txt', header=None)







accuracy = accuracy_score

recall = recall_score(y_train, y_pred , average="binary")

precision = precision_score(y_train, y_pred , average="binary")

f1 = f1_score(y_train, y_pred, average="binary")



cnn.load_weights("cnnresults/checkpoint-317.hdf5")

cnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

loss, accuracy = cnn.evaluate(X_test, y_test)





y_pred = cnn.predict_classes(X_test)

np.savetxt('res/expected3.txt', y_test, fmt='%01d')

np.savetxt('res/predicted3.txt', y_pred, fmt='%01d')



accuracy = accuracy_score(y_test, y_pred)

recall = recall_score(y_test, y_pred , average="binary")

precision = precision_score(y_test, y_pred , average="binary")

f1 = f1_score(y_test, y_pred, average="binary")



print("confusion matrix")

print("----------------------------------------------")

print("accuracy")

print("%.6f" %accuracy)

print("racall")

print("%.6f" %recall)

print("precision")

print("%.6f" %precision)

print("f1score")

print("%.6f" %f1)

cm = metrics.confusion_matrix(y_test, y_pred)
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
cnn.fit(X_train, y_train, callbacks=[checkpointer,csv_logger])

cnn.save("cnnlayer_model.hdf5")

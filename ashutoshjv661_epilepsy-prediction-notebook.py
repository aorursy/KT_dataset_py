from __future__ import print_function
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from keras import backend as K

import random
import numpy as np
import pandas as pd

import scipy.io
from scipy.signal import spectrogram
from scipy.signal import resample
import matplotlib.pyplot as plt
#Visualizing an example:
interictal_tst = '../input/seizure-prediction/Patient_1/Patient_1/Patient_1_interictal_segment_0001.mat'
preictal_tst = '../input/seizure-prediction/Patient_1/Patient_1/Patient_1_preictal_segment_0001.mat'
interictal_data = scipy.io.loadmat(interictal_tst)
preictal_data = scipy.io.loadmat(preictal_tst)


interictal_array = interictal_data['interictal_segment_1'][0][0][0]
preictal_array = preictal_data['preictal_segment_1'][0][0][0]
#EXTRA
print(type(preictal_data['preictal_segment_1']) , preictal_data['preictal_segment_1'][0][0][0].shape)

# Creating training and testing data
all_X = []
all_Y = []

types = ['Patient_1_interictal_segment', 'Patient_1_preictal_segment']

for i,typ in enumerate(types):
    # Looking at 18 files for each event for a balanced dataset
    for j in range(18):
        fl = '../input/seizure-prediction/Patient_1/Patient_1/{}_{}.mat'.format(typ, str(j + 1).zfill(4))
        data = scipy.io.loadmat(fl)
        k = typ.replace('Patient_1_', '') + '_'
        #downsampling the data
        d_array = data[k + str(j + 1)][0][0][0]
        secs = len(d_array[14])/5000 # Number of seconds in signal X
        samps = secs*500     # Number of samples to downsample
        dsample_array = scipy.signal.resample(d_array[14],300000)
        
        lst = list(range(300000))  # 3000000  datapoints initially
        for m in lst[::2000]: #5000 initial 
            # Create a spectrogram every 2 second
            p_secs = dsample_array[m:m+2000]    #d_array[0][m:m+15000]
            p_f, p_t, p_Sxx = spectrogram(p_secs, fs=500, return_onesided=False)
            p_SS = np.log1p(p_Sxx)
            arr = p_SS[:] / np.max(p_SS)
            all_X.append(arr)
            all_Y.append(i)
print(all_X[0].shape)

print(len(all_X))
print(len(all_Y))
# Shuffling the data
dataset = list(zip(all_X, all_Y))
random.shuffle(dataset)
all_X,all_Y = zip(*dataset)
print(len(all_X))
print(all_X[0].shape)
# Splitting data into train/test, leaving only 600 samples for testing
x_train = np.array(all_X[:4800])
y_train = np.array(all_Y[:4800])
x_test = np.array(all_X[4800:])
y_test = np.array(all_Y[4800:])
print(x_train[0].shape)
print(len(x_test))
print(type(x_test))
print("----------")
print(y_train.shape)
print(len(y_train))

batch_size = 128
num_classes = 2
epochs = 100
img_rows, img_cols = 256,8
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Formatting the labels for training
y_train = tf.keras.utils.to_categorical(y_train, num_classes) 
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

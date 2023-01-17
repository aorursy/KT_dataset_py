!pip install pandas==0.24.2
import pandas as pd

import glob

import cv2

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential, Model

from keras.layers import BatchNormalization, Input,Conv2D, Activation, MaxPooling2D,MaxPooling2D,AveragePooling2D, Dense, GlobalAveragePooling2D

from keras import optimizers

from keras.layers import Dropout, Flatten, Concatenate

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler

import keras_resnet.models

import keras

from keras import callbacks

from keras.models import model_from_json

import h5py

tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
""""path='\\project\\asl_alphabet_train'

all_files=glob.glob(path + "\\*.csv")

li=0

for filename in all_files:

    print(filename)

    df = pd.read_csv(filename, index_col=None,header=None, dtype='uint8')

    if(li==0):        

        frame=df

        li=li+1

        print(frame.shape)    

        print(frame.head())

    else:    

        frame = pd.concat([frame,df], ignore_index=True,sort=False)

        print(frame.shape)    

        print(frame.head())    

#frame = pd.concat(li, axis=0, ignore_index=True)

#frame = pd.concat([] ignore_index=True)

frame.to_pickle('.\\pixels_pickle.pkl')"""
"""#frame= pd.read_pickle('')

print(frame.shape)

print(frame.head())  

frame = frame.sample(frac=1).reset_index(drop=True)

print(frame.head())

print(frame.shape)"""
#frame.to_pickle('.\\pixels_pickle.pkl')
frame = pd.read_pickle("/kaggle/input/pixels_pickle.pkl")

data = frame.values

#data_cv=df_crossval.values

print(np.shape(data))#,np.shape(data_cv))

y = data[:, 0]

#y_test=data_cv[:, 0]

print(frame.head())
pixels = data[:, 1:40001]

print(np.shape(pixels))

print(len(pixels[0]))

print(pixels[10][10])
X=pixels

print(np.shape(X))
np.save('facial_data_X', X)

np.save('facial_labels', y)       
from matplotlib import pyplot as plt

%matplotlib inline
x =X
for ix in range(10):

    plt.figure(ix)

    plt.imshow(x[ix].reshape((200, 200)), interpolation='none', cmap='gray')

plt.show()
X_train = x[0:57000,:]

Y_train = y[0:57000]

print(X_train.shape , Y_train.shape)

X_crossval = x[57000:58000]#x_t[:,:]

Y_crossval = y[57000:58000]#y_t[:]

print (X_crossval.shape , Y_crossval.shape)
X_train = X_train.reshape((X_train.shape[0],200, 200,1))

X_crossval = X_crossval.reshape((X_crossval.shape[0],200, 200,1))

print(X_train.shape,X_crossval.shape)
from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

y_1 = np_utils.to_categorical(Y_train)

y_2 = np_utils.to_categorical(Y_crossval)
model = Sequential()



    # 1 - Convolution

model.add(Conv2D(64,(3,3), padding='same', input_shape=(200, 200,1)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.50))



    # 2nd Convolution layer

model.add(Conv2D(128,(5,5), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.50))



# 3rd Convolution layer

model.add(Conv2D(512,(3,3), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.50))



# 4th Convolution layer

model.add(Conv2D(512,(3,3), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.50))





# Flattening

model.add(Flatten())



# Fully connected layer 1st layer

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.50))





# Fully connected layer 2nd layer

model.add(Dense(512))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.50))



model.add(Dense(30, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_1,

              batch_size=16,

              epochs=10,

              verbose=2,

              validation_split=0.1111)



score = model.evaluate(X_crossval, y_2, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])

model_json = model.to_json()

with open('model_dropout.json', 'w') as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights('modelfer_dropout50.h5')

print('Saved model to disk')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Any results you write to the current directory are saved as output.

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#Loading Data
(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()
#defining Parameters
num_classes = 10
batch_size = 128
epoch = 24
img_rows, img_cols = 28,28
#Deal with format issues between different backends. Some put the no. of channels in the image before the width and height.
if K.image_data_format() == 'channels_first':
    X_train=X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
    X_test =X_test.reshape(X_test.shape[0],1,img_rows,img_cols)
    input_shape=(1,img_rows,img_cols)
else:
    X_train=X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
    X_test =X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
    input_shape=(img_rows,img_cols,1)
#Convert and  scale the test and training data. Bring the scale from 0-255 to 0-1.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#Convert class vectors to binary class matrices using One-hot encoding
y_train = keras.utils.to_categorical(y_train,num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes=num_classes)
#Define the model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2,2)))   #Removing MaxPooling layer: Add accuracy but reduces training speed
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
model.summary()
#Adding Callback for early stopping
from keras.callbacks import EarlyStopping
my_callback=[EarlyStopping(monitor='val_acc',patience=5,mode=max)]

#define compile to minimize categorical loss
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.adadelta(),
              metrics=['accuracy'])
#Train the model and test/validate the mode with the test data after each cycle(epoch) through the training data
#Return history of loss and accuracy for each epoch
hist= model.fit(X_train,y_train,
               batch_size=batch_size,
               epochs=epoch,verbose=1,callbacks=my_callback,              
               validation_data=(X_test,y_test))
#score = model.evaluate(X_test,y_test,verbose=0)
#print('Test loss: ', score[0])
#print('Test accuracy', score[1])

#hist.history.keys()

epoch_list = list(range(1,len(hist.history['acc'])+1))  #Values for x axis[1,2,3,4...# of epochs]
plt.plot(epoch_list, hist.history['acc'],epoch_list,hist.history['val_acc'])
plt.legend(('Training accuracy','Validation Accuracy'))
plt.show()


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.datasets import cifar10

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import tarfile
import sys
import pickle
print(os.listdir("../input/cifar-10-python/cifar-10-batches-py"))

import keras
from keras import backend   as K
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D,Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.
def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels
train_num = 50000
train_x = np.zeros(shape=(train_num,3,32,32))
train_y = np.zeros(shape=(train_num))

test_num = 10000
test_x = np.zeros(shape=(test_num,3,32,32))
test_y = np.zeros(shape=(test_num))
def load_data():
    for i in range(1,6):
        begin = (i-1)*10000
        end = i*10000
        train_x[begin:end,:,:,:],train_y[begin:end] = load_batch("../input/cifar-10-python/cifar-10-batches-py/data_batch_"+str(i))
    
    test_x[:],test_y[:] = load_batch("../input/cifar-10-python/cifar-10-batches-py/test_batch")
load_data()
test_y[1:10]
if K.image_data_format() == 'channels_last':
    print("channels_last")
    test_x = test_x.transpose(0, 2, 3, 1)
    train_x = train_x.transpose(0, 2, 3, 1)
else:
    print("channels_first")

train_x.shape
train_y.shape
train_y[0:10]
test_x.shape
test_y.shape
#normalization
train_x = train_x/255.0
test_x = test_x/255.0
train_y = to_categorical(train_y,10)
test_y = to_categorical(test_y,10)
train_x.shape
train_y.shape
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=train_x.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
batch_size = 32
epochs = 100
history = model.fit(train_x,train_y,batch_size,epochs,validation_data=(test_x,test_y),shuffle=True)
#histroy.history
fig = plt.plot(history.history["acc"],label = "train", color='green')
plt.plot(history.history["val_acc"],label = "test", color='red')
plt.legend(loc='upper left')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("accuracy by epochs")
plt.show()

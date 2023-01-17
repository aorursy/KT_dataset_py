# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(10)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, InputLayer, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import matplotlib.image as mpim
import matplotlib.pyplot as plt
import gc

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/fashion-mnist_train.csv')
test_data = pd.read_csv('../input/fashion-mnist_test.csv')

img_size = 28
input_shape = (img_size, img_size, 1)
#prepare data: X_train is the numpy array of images, y_train is the numpy array of labels
X_train = np.array(train_data.iloc[:,1:])
y_train = to_categorical(np.array(train_data.iloc[:,0]))
del train_data
gc.collect()
#preprocess input
X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 1)
X_train = X_train.astype('float32')
X_train /= 255
#2 Convolutional Layers
def train_2_conv():
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), activation='relu', bias_initializer='RandomNormal', kernel_initializer='random_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
#3 Convolutional Layers
def train_3_conv(shape=(28, 28, 1), num_classes=10):
    X_input = Input(shape=shape)
    X = Conv2D(64,kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape)(X_input)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.25)(X)
    X = Conv2D(512, (3, 3), activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.3)(X)
    X = Conv2D(128, (3, 3), activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(num_classes, activation='softmax')(X)
    model = Model(inputs=X_input, outputs=X, name="CNN")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#VGG6 model
def VGG_16():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
model = train_2_conv()
training = model.fit(X_train, y_train, batch_size = 256, epochs = 30, shuffle = False, validation_split = 0.1, verbose = 2)
#prepare test data
X_test = np.array(test_data.iloc[:,1:])
y_test = np.array(test_data.iloc[:,0])

#preprocess input for model
X_test = X_test.reshape(X_test.shape[0], img_size, img_size, 1)
X_test = X_test.astype('float32')
X_test /= 255
#evaluate function checks the accuracy and loss of test set
from sklearn.metrics import accuracy_score
pred = model.predict(X_test)
# convert predicions from categorical back to 0...9 digits
pred_digits = np.argmax(pred, axis=1)
print("Test Set Accuracy:")
print(accuracy_score(y_test, pred_digits))
def analysis_plot(training):
    loss = training.history['loss']
    val_loss = training.history['val_loss']
    acc = training.history['acc']
    val_acc = training.history['val_acc']

    # loss plot
    plt.plot(loss)
    plt.plot(val_loss, 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend(["Training", "Validation"])

    plt.show()

    # accuracy plot
    plt.plot(acc)
    plt.plot(val_acc, 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Accuracy')
    plt.legend(['Training', 'Validation'], loc=4)
    plt.show()
analysis_plot(training)
model = train_3_conv()
training = model.fit(X_train, y_train, batch_size = 256, epochs = 30, shuffle = False, validation_split = 0.1, verbose = 2)
pred = model.predict(X_test)
# convert predicions from categorical back to 0...9 digits
pred_digits = np.argmax(pred, axis=1)
print("Test Set Accuracy:")
print(accuracy_score(y_test, pred_digits))
analysis_plot(training)
model = VGG_16()
training = model.fit(X_train, y_train, batch_size = 256, epochs = 25, shuffle = False, validation_split = 0.1, verbose = 2)
pred = model.predict(X_test)
# convert predicions from categorical back to 0...9 digits
pred_digits = np.argmax(pred, axis=1)
print("Test Set Accuracy:")
print(accuracy_score(y_test, pred_digits))
analysis_plot(training)
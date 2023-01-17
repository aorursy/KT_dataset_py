# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.datasets import mnist

from keras.models import Sequential

from keras.layers.core import Dense,Activation,Dropout,Flatten

from keras.optimizers import SGD,RMSprop,Adam

from keras.utils import np_utils

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras import backend as K



np.random.seed(1671) # for reproducibility





import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

class LeNet:

    def build(input_shape,classes):

        model=Sequential()

        model.add(Conv2D(20,kernel_size=5,padding="same",input_shape=input_shape))

        model.add(Activation("relu"))

        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        model.add(Conv2D(50,kernel_size=5,padding="same"))

        model.add(Activation("relu"))

        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        model.add(Flatten())

        model.add(Dense(500))

        model.add(Activation("relu"))

        model.add(Dense(classes))

        model.add(Activation("softmax"))

        return model


NB_EPOCH=10

BATCH_SIZE=128

VERBOSE=1

NB_CLASSES=10 # number of outputs = number of digits

IMG_ROWS,IMG_COLS=28,28 # input image dimensions

#OPTIMIZER=SGD() # SGD Optimizer

#OPTIMIZER=RMSprop()

OPTIMIZER=Adam()

VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION

INPUT_SHAPE=(1,IMG_ROWS,IMG_COLS)
def load_data(path):

    with np.load(path) as f:

        x_train, y_train = f['x_train'], f['y_train']

        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)

#data Shuffled and split between train and test sets

(X_train,y_train),(X_test,y_test)=load_data("../input/mnist.npz")

K.set_image_dim_ordering("th")
X_train=X_train.astype('float32')

X_test=X_test.astype('float32')

#normalize

X_train/=255

X_test/=255

#we need a 60K*[1*28*28] shape as inout to the CONVET

X_train=X_train[:,np.newaxis,:,:]

X_test=X_test[:,np.newaxis,:,:]


print(X_train.shape[0],'train samples')

print(X_test.shape[0],'test samples')



#Convert class Vectors to binary class matrices



Y_train=np_utils.to_categorical(y_train,NB_CLASSES)

Y_test=np_utils.to_categorical(y_test,NB_CLASSES)

#initilize the optimizer and model

model=LeNet.build(input_shape=INPUT_SHAPE,classes=NB_CLASSES)

model.compile(loss="categorical_crossentropy",optimizer=OPTIMIZER,metrics=["accuracy"])

history=model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=NB_EPOCH,verbose=VERBOSE,validation_split=VALIDATION_SPLIT)

score=model.evaluate(X_test,Y_test,verbose=VERBOSE)

print("Test Score",score[0])

print("Test accuracy",score[1])

print(history.history.keys())


plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train','test'],loc='upper left')

plt.show()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train','test'],loc='upper left')

plt.show()

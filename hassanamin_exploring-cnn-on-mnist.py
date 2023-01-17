# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils as kutils
from sklearn.model_selection import train_test_split


##(X_train, y_train), (X_test, y_test) = mnist.load_data()
np.random.seed(1671) # for reproducibility
# network and training parameters
NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # number of outputs = number of digits
RESHAPED = 784
OPTIMIZER = SGD() # SGD optimizer, explained later in this chapter
N_HIDDEN = 128
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION
img_rows=28
img_cols=28
# data: shuffled and split between train and test sets
#

def load_dataset(train_path='../input/train.csv',test_path='../input/test.csv'):
    #global train,test,trainX,trainY,nb_classes
    
    train = pd.read_csv(train_path).values # produces numpy array
    test  = pd.read_csv(test_path).values # produces numpy array
    print("Train Shape :",train.shape)
    print("Test Shape :",test.shape)
    trainX = train[:, 1:].reshape(train.shape[0], img_rows,img_cols,1)
    trainX = trainX.astype(float)
    trainX /= 255.0
    trainY = kutils.to_categorical(train[:, 0])
    nb_classes = trainY.shape[1]
    print("TrainX Shape : ",trainX.shape)
    print("Trainy shape : ",trainY.shape)
    testX = test.reshape(test.shape[0], img_rows,img_cols,1)
    testX = testX.astype(float)
    testX /= 255.0
    testY = kutils.to_categorical(test[:, 0])
    return trainX,trainY,testX,testY,nb_classes

X,y,testX,testY,nb_classes=load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(type(X_train))
#img_rows, img_cols = X_train[0].shape[0], X_train[0].shape[1]
#X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_acc', patience=5)]
batch_size = 128
n_epochs = 2
model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_split=0.2, callbacks=callbacks)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Extract predictions
preds = model.predict(X_test)

n_examples = 10
plt.figure(figsize=(15, 15))
for i in range(n_examples):
    ax = plt.subplot(2, n_examples, i + 1)
    plt.imshow(X_test[i, :, :, 0], cmap='gray')
    plt.title("Label: {}\nPredicted: {}".format(np.argmax(y_test[i]), np.argmax(preds[i])))
    plt.axis('off')
    
plt.show()
plt.figure(figsize=(15, 15))

j=1
for i in range(len(y_test)):
    if(j>10):
        break
    label = np.argmax(y_test[i])
    pred = np.argmax(preds[i])
    if label != pred:        
        ax = plt.subplot(2, n_examples, j)
        plt.imshow(X_test[i, :, :, 0], cmap='gray')
        plt.title("Label: {}\nPredicted: {}".format(label, pred))
        plt.axis('off')
        j+=1
plt.show()
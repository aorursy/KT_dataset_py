# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv").values
def convert_to_onehot(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b
X = np.reshape(train[:, 1:], (-1, 28, 28, 1))
Y = train[:, 0]

img_num = 6001
plt.imshow(X=X[img_num].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.title("Y[" + str(img_num) + "] = " + str(Y[img_num]))

print("Y[" + str(img_num) + "] = " + str(Y[img_num]))
Y = convert_to_onehot(Y)
print("One hot encoded Y[" + str(img_num) + "] = " + str(Y[img_num]))
print()

X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.1, random_state=42)
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_dev shape: " + str(X_dev.shape))
print("Y_dev shape: " + str(Y_dev.shape))
def digitRecogModel(input_shape):
    X_input = Input(input_shape)
    
    X = Conv2D(filters=16, kernel_size=3)(X_input)
    X = Activation('relu')(X)
    X = BatchNormalization(axis=3)(X)
    X = Dropout(0.2)(X)
    X = MaxPooling2D(pool_size=2)(X)
    
    X = Conv2D(filters=32, kernel_size=4)(X)
    X = Activation('relu')(X)
    X = BatchNormalization(axis=3)(X)
    X = Dropout(0.2)(X)
    X = MaxPooling2D(pool_size=2)(X)
    
    X = Conv2D(filters=64, kernel_size=4)(X)
    X = Activation('relu')(X)
    X = BatchNormalization(axis=3)(X)
    X = Dropout(0.2)(X)
    X = MaxPooling2D(pool_size=2)(X)
    
    X = Flatten()(X)
    X = Dense(64, activation='relu')(X)
    X = BatchNormalization(axis=1)(X)
    X = Dropout(0.2)(X)
    X = Dense(32, activation='relu')(X)
    X = BatchNormalization(axis=1)(X)
    X = Dropout(0.2)(X)
    X = Dense(16, activation='relu')(X)
    X = BatchNormalization(axis=1)(X)
    X = Dropout(0.2)(X)
    X = Dense(10, activation='sigmoid')(X)
    
    model = Model(inputs=X_input, outputs=X)
    return model
model = digitRecogModel(X_train[0].shape)
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
start = time.time()
model.fit(x=X_train, y=Y_train, epochs=100, batch_size=20)
print("Training Time: " + str(time.time() - start))
preds = model.evaluate(x=X_dev, y=Y_dev)

print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
test = pd.read_csv('../input/test.csv').values
X_test = np.reshape(test, (-1, 28, 28, 1))
results = model.predict(x=X_test)
results = np.argmax(results, axis=1)
img_num = 3
plt.imshow(X=X_test[img_num].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.title('Prediction is ' + str(results[img_num]))
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
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
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import tensorflow as tf
print(tf.__version__)
test= pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
test.columns, train.columns
y_train = train.iloc[:,0].values.astype('int32') 
X_train = (train.iloc[:,1:].values).astype('float32') 

X_test = test.values.astype('float32')
X_train.shape
X_train = X_train.reshape(X_train.shape[0], 28, 28)
X_test = X_test.reshape(X_test.shape[0], 28, 28)
X_train.shape
f, axarr = plt.subplots(5, sharex=True)
for i in range(5):
    print(y_train[i])
    axarr[i].imshow(X_train[i])
#     Use Below line for black and white images
#     axarr[i].imshow(X_train[i], cmap=plt.cm.binary)
X_train[0]
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)
f, axarr = plt.subplots(5, sharex=True)
for i in range(5):
    print(y_train[i])
    axarr[i].imshow(X_train[i])
#     Use Below line for black and white images
#     axarr[i].imshow(X_train[i], cmap=plt.cm.binary)
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D
model = Sequential()
model.add(Flatten()) # Since we want a flat input
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',
             metrics=['accuracy'],
             loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)
predictions = model.predict(X_test)
val_loss, val_acc = model.evaluate(X_train, y_train)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)
predictions
print(np.argmax(predictions[0]))
plt.imshow(X_test[0],cmap=plt.cm.binary)
plt.show()
predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions})
submissions.to_csv("sub.csv", index=False, header=True)
submissions.head()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sbn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/fashion-mnist_train.csv')

test_data = pd.read_csv('../input/fashion-mnist_test.csv')
train_data.head() # 784 columns for 28 x 28 pixels and 1 column for response variable(label)
X_train = train_data.iloc[:, 1:].values # 0th colm is response variable

y_train = train_data.iloc[:, 0].values # response variable



X_test = test_data.iloc[:, 1:].values

y_test = test_data.iloc[:, 0].values
print(X_train.shape)

print(y_train.shape)
# looking at some images



import random

i = random.randint(1, 60000) # to select a random  row

plt.imshow(X_train[i, :].reshape((28, 28))) # converting 784 pixels into 28 x 28 matrix and viewing it as an image
print (y_train[i])
# Normalizing the data

X_train = X_train/255



X_test = X_test/255
X_train.dtype
# splitting into training and validation set

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))

X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))

X_val = X_val.reshape(X_val.shape[0], *(28, 28, 1))
X_train.shape
# converting output to categorical data

# from keras.utils import to_categorical



# y_train = to_categorical(y_train)

# y_test = to_categorical(y_test)

# y_val = to_categorical(y_val)
y_train.shape 
# Training using CNN

from keras.models import Sequential

from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard
cnn = Sequential() # model has layers in sequence



# convolutional layer

cnn.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu')) # convoluting

cnn.add(MaxPooling2D(pool_size=(2,2))) # pooling



cnn.add(Dropout(0.25)) # to prevent overfitting





cnn.add(Flatten()) # flattening into 1D set.

# This is input layer of the Neural Network



cnn.add(Dense(units=32, activation='relu')) # first hidden layer with 32 units

# this model has only one hidden layer



cnn.add(Dense(units=10, activation='sigmoid')) # Output layer
cnn.summary()
cnn.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])



cnn.fit(X_train, y_train, batch_size=512, epochs=50, verbose=1, validation_data=(X_val, y_val))
eval_score = cnn.evaluate(X_test, y_test)

print('Loss:', eval_score[0], '\nAccuracy:', eval_score[1])
from sklearn.metrics import confusion_matrix, classification_report



print("CNN Model-1:\n")

y_pred = cnn.predict_classes(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(14, 10))

sbn.heatmap(cm, annot=True)



print(classification_report(y_test, y_pred, target_names=["Class {}".format(i) for i in range(10)]))
# Comparing results with actual label

fig, axes = plt.subplots(5, 5, figsize=(12, 12))

axes = axes.ravel()



for i in np.arange(0, 5*5):

    x = random.randint(0, 10000)

    

    axes[i].imshow(X_test[x].reshape(28, 28))

    

    axes[i].set_title("Predicted: {}, Actual: {}".format(y_pred[x], y_test[x]))

    

    axes[i].axis('off')



plt.subplots_adjust(wspace=0.5)



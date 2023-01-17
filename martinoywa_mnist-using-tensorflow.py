# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# training dataset

train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
# testing dataset

test = pd.read_csv("../input/test.csv")

print(test.shape)

test.head()
X_train = train.iloc[:, 1:].values.astype("float") # features only. i.e from pixel0

y_train = train.iloc[:,0].values.astype('int') # target labels only. i.e label

X_test = test.values.astype('float') # test containing features
# preview the images

plt.figure(figsize=(12,10))

x, y = 10, 4

for i in range(40):  

    plt.subplot(y, x, i+1)

    plt.imshow(X_train[i].reshape((28,28)),interpolation='nearest')

plt.show()
# Normalization

X_train = X_train / 255.0

X_test = X_test / 255.0
print('X_train shape:', X_train.shape)

print(X_train.shape[0], 'train samples')

print(X_test.shape[0], 'test samples')
# creating 28 by 28 images

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
import tensorflow as tf
# my model

model = tf.keras.Sequential([

        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),

        tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Conv2D(64, 3, activation="relu"),

        tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Conv2D(128, 3, activation="relu"),

        tf.keras.layers.Dropout(.5),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(100, activation="relu"),

        tf.keras.layers.Dense(10, activation="softmax"),

    ])
model.summary()
# compiling the model

model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# training the model

model.fit(X_train, y_train, epochs=150)
# predictions

y_predict = model.predict_classes(X_test)

y_predict
# creating the submission file

submission = pd.DataFrame({"ImageId":[i+1 for i in range(len(X_test))],

                           "Label": y_predict})

submission.head()
submission.to_csv("submission_5.csv", index=False, header=True)
# saving the model

model.save("MNIST_MODEL.h5")
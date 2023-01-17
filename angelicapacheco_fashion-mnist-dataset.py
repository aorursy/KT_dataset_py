# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

%matplotlib inline

# Any results you write to the current directory are saved as output.
#Load train and test sets

df_train = pd.read_csv("../input/mnist-dataset-train/fashion-mnist_train.csv")

df_test= pd.read_csv("../input/mnist-dataset-test/fashion-mnist_test.csv")
df_test.head()
df_train.shape
df_test.shape
#Split into X and y and convert to categorical 

from keras.utils import to_categorical

X_train = np.array(df_train.iloc[:, 1:])

X_test = np.array(df_test.iloc[:, 1:])

y_train = to_categorical(np.array(df_train.iloc[:, 0]))

y_test = to_categorical(np.array(df_test.iloc[:, 0]))

#Let´s see couple of products

plt.imshow(X_train[1229].reshape((28,28)), cmap='Greys_r')
plt.imshow(X_train[6000].reshape((28,28)), cmap='Greys_r')
#Conversion of the data 

#It is common to use 32-bit precision when training a neural network.

#255 is the max. value of a byte, dividing by 255 will ensure that the input features are scaled between 0.0 and 1.0



X_train = X_train.astype('float32') / 255

y_train = y_train.astype('float32') / 255
#Create model with keras

from keras import models

from keras import layers

from keras import optimizers





model = models.Sequential()

model.add(layers.Dense(40, activation="relu", input_shape=(784,)))

model.add(layers.Dense(20, activation="relu"))

model.add(layers.Dense(20, activation="relu"))

model.add(layers.Dense(10, activation="softmax"))





#Define an optimizer, compile and see a summary of the model



model.compile(optimizer='adam',

               loss='categorical_crossentropy',

               metrics=['accuracy'])

model.summary()

#Fit the model to the data

model.fit(X_train, y_train, epochs=8, batch_size=100)



#Evaluate model

score = model.evaluate(X_test, y_test, verbose=0)

#Let´s see the results

print('Accuracy of the model: ',score[1])

print('Loss of the model: ',score[0])
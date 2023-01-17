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
import tensorflow as tf
mnist_input_train = pd.read_csv('../input/train.csv')

mnist_input_test = pd.read_csv('../input/test.csv')
print(mnist_input_train.info())
X_train = np.array(mnist_input_train)[:, 1:785]

y_train = np.array(mnist_input_train)[:, 0]

print(X_train.shape)

print(y_train.shape)



X_test = np.array(mnist_input_test)[:, :]

print(X_test.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train.shape
input_shape = (28, 28, 1)

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')
X_train /= 255

X_test /= 255
print('x_train shape:', X_train.shape)

print('Number of images in x_train', X_train.shape[0])

print('Number of images in x_test', X_test.shape[0])
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
model = Sequential()

model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation=tf.nn.relu))

model.add(Dropout(0.2))

model.add(Dense(10,activation=tf.nn.softmax))
y_train.shape
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy', 

              metrics=['accuracy'])

model.fit(x=X_train,y=y_train, epochs=10)
import matplotlib.pyplot as plt

%matplotlib inline
image_index = 4444

plt.imshow(X_test[image_index].reshape(28, 28),cmap='Greys')

pred = model.predict(X_test[image_index].reshape(1, 28, 28, 1))

print(pred.argmax())
X_test.shape
predictions = model.predict_classes(X_test, verbose=1)

pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),

              "Label":predictions}).to_csv("submission.csv",

                                           index=False,

                                           header=True)
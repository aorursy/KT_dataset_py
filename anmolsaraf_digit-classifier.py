# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D

import cv2
mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
mnist_train.head()
y_train=mnist_train['label'].to_numpy()

y_test=mnist_test['label'].to_numpy()

print("y_train_shape=", y_train.shape)

y_train.resize(60000,1)

print("new_y_train_shape=", y_train.shape)

y_test.resize(y_test.shape[0], 1)

print("new_y_test_shape=", y_train.shape)
x_train=mnist_train.loc[:, mnist_train.columns!='label']

x_test=mnist_test.loc[:, mnist_test.columns!='label']

x_train=x_train.to_numpy()

x_test=x_test.to_numpy()
x_train1=x_train.reshape(60000,-1)

x_test1=x_test.reshape(x_test.shape[0],-1)

x_train2=x_train1.reshape(60000,28,28,1)

x_test2=x_test1.reshape(x_test1.shape[0],28,28,1)

x_train3=x_train2.astype('float32')/255

x_test3=x_test2.astype('float32')/255
plt.imshow(x_train2[400].squeeze(),cmap='Greys')

print(y_train[400])
input_shape=(28,28,1)

model = Sequential() 

model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(256,activation = tf.nn.relu))

model.add(Dropout(0.2))

model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

model.fit(x= x_train3 , y = y_train , batch_size= 2000, epochs= 10)#
model.evaluate(x_test3, y_test)
print("predicted=", np.argmax(model.predict(x_test3[510:511]), axis=1))

print("actual=", y_test[510:511].squeeze())

plt.imshow(x_test2[510].squeeze(), cmap='Greys')
test_image=cv2.imread(r"../input/sevennimage/Seven.png")

test_image=cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)

test_image=np.invert(test_image)

plt.imshow(test_image.squeeze(), cmap='Greys')

type(test_image)

#ti=cv2.resize(test_image, (28,28), interpolation=cv2.INTER_AREA)

#plt.imshow(ti, cmap="Greys")
ti=(test_image.astype("float32"))/255

plt.imshow(ti.squeeze(), cmap='Greys')

ti.resize(28,28,1)

ti.shape

#plt.imshow(ti.squeeze(), cmap='Greys')

images=np.array([ti,ti])

images.shape
print("Predicted number is %s." %np.argmax(model.predict(images[0:1]), axis=1).squeeze())
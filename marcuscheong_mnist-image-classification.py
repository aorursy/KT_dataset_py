# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf



(x_train,y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
print("--------------------")

print("x_train: " , x_train.shape)

print("--------------------")

print("y_train: " , y_train.shape)

print("--------------------")

print("x_test: " , x_test.shape)

print("--------------------")

print("y_test: " , y_test.shape)

print("--------------------")

import matplotlib.pyplot as plt



sample_img = x_train[0]



TRAINSIZE = x_train.shape[0]



TESTSIZE = x_test.shape[0]



IMGSIZE = sample_img.shape[1]



plt.imshow(sample_img)

print(TRAINSIZE, TESTSIZE, IMGSIZE)
x_train = x_train.reshape(TRAINSIZE, IMGSIZE,IMGSIZE, 1)

x_test = x_test.reshape(TESTSIZE, IMGSIZE, IMGSIZE, 1)



#Rescale Data between 0-1

x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test  = tf.keras.utils.normalize(x_test, axis=1)





from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



model = Sequential()

INPUT_SHAPE = (IMGSIZE, IMGSIZE, 1)



#First Layer

model.add(Conv2D(28,kernel_size = (3,3), input_shape = INPUT_SHAPE))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())



#Second Layer

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.2))



#Output Layer

model.add(Dense(10, activation = "softmax"))
model.compile(optimizer = "adam",

             loss ="sparse_categorical_crossentropy",

             metrics = ["accuracy"])



model.fit(x_train,y_train, epochs = 20)
model.evaluate(x_test, y_test)
test_img = x_test[678]

test_img = test_img.reshape(1,28,28,1)

predict_ = model.predict(test_img)[0]



for i in range(len(predict_)):

    if(predict_[i] == max(predict_)):

        print("The number written is: ", i)

        

plt.imshow(test_img.reshape(28,28), cmap='gray')
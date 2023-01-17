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
!pip install idx2numpy
import numpy as np
import gzip
import idx2numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras

print(os.listdir('/kaggle/input/'))
train_X = idx2numpy.convert_from_file('/kaggle/input/train-images.idx3-ubyte').reshape(-1,28,28,1)
train_y = idx2numpy.convert_from_file('/kaggle/input/train-labels.idx1-ubyte')
test_X = idx2numpy.convert_from_file('/kaggle/input/t10k-images.idx3-ubyte').reshape(-1,28,28,1)
test_y = idx2numpy.convert_from_file('/kaggle/input/t10k-labels.idx1-ubyte')

train_X.shape, train_y.shape, test_X.shape, test_y.shape
with tf.compat.v1.Session() as sess:
    train_y = sess.run(tf.one_hot(train_y,10))
    test_y = sess.run(tf.one_hot(test_y,10))
    sess.close()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
model1 = Sequential()

model1.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model1.add(Conv2D(32,kernel_size=3,activation='relu'))
model1.add(Flatten())
model1.add(Dense(10,activation='softmax'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model1.fit(train_X, train_y, validation_data=(test_X, test_y), batch_size=500, epochs=3)
model1.predict(test_X[:4])
np.argmax(model1.predict(test_X[:4]),1)
model1.save('/kaggle/working/MNIST_CNN.h5')

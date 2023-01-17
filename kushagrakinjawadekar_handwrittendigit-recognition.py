# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/digit-recognizer'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

print(train.shape)

print(test.shape)
y_train = train['label']

X_train = train.drop(labels = ["label"],axis = 1)
import matplotlib.pyplot as plt

%matplotlib inline
# Normalize the data

X_train = X_train / 255.0

test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
print(X_train.shape)

print(y_train.shape)

print(test.shape)
import tensorflow as tf

print(tf.__version__)
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

y_train = to_categorical(y_train, num_classes = 10)
plt.imshow(X_train[100][:,:,0])
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size= 0.2,random_state = 101)

from tensorflow import keras

from tensorflow.keras import layers, models

model = models.Sequential()

model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Dropout(0.25))





model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(layers.Dropout(0.25))





model.add(layers.Flatten())

model.add(layers.Dense(256, activation = "relu"))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation = "softmax"))
model.summary()
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size = 100, epochs = 10, validation_data = (X_test, y_test))
results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist.csv",index=False)
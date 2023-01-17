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
train_data = pd.read_csv('../input/digit-recognizer/train.csv')

test_data = pd.read_csv('../input/digit-recognizer/test.csv')
y = train_data.pop('label')

x = train_data

# normalization

x = x / 255.0

test_data = test_data / 255.0

# Converting dataframe into arrays

x = np.array(x)

y = np.array(y)

# shaping

x = x.reshape(-1,28,28,1)

# enci=oding

from keras.utils.np_utils import to_categorical

y = to_categorical(y, 10)
test_data=test_data.values
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



cnn = keras.models.Sequential()

cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1))) 

cnn.add(keras.layers.MaxPool2D(pool_size = 2, strides=2))

cnn.add(keras.layers.Flatten())

cnn.add(keras.layers.Dense(128, activation='relu'))

cnn.add(keras.layers.Dense(10,activation='softmax'))
cnn.compile(  optimizer= 'Adam', loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
history = cnn.fit(x, y, shuffle= True, epochs=20, validation_split=0.2)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(history.history['accuracy'],label = 'ACCURACY')

plt.plot(history.history['val_accuracy'],label = 'VALIDATION ACCURACY')

plt.legend()
plt.plot(history.history['loss'],label = 'TRAINING LOSS')

plt.plot(history.history['val_loss'],label = 'VALIDATION LOSS')

plt.legend()
test_data.shape
test_data = test_data.reshape(-1, 28, 28 , 1).astype('float32')

import numpy

res = cnn.predict(test_data)

res = numpy.argmax(res,axis = 1)

res = pd.Series(res, name="Label")

submission = pd.concat([pd.Series(range(1 ,28001) ,name = "ImageId"),   res],axis = 1)

submission.to_csv("My_submission.csv",index=False)
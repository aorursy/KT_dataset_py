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
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv', index_col=False)
train.head()
train.shape
train.iloc[:,0]
x_train = train.iloc[:, 1:].values.astype('float32')
x_train.shape
y_train = train.iloc[:, 0].values.astype('int32')
y_train
y_train.shape
import matplotlib.pyplot as plt
img = x_train[1].copy()

img = img.reshape(28, 28)
plt.imshow(img)
y_train[1]
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

x_test = test.values.astype('float32')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit_transform(x_train)
scaler.fit_transform(x_test)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train.shape, x_test.shape
import tensorflow as tf

from tensorflow import keras

from keras.layers import Conv2D, MaxPooling2D

from keras.models import Sequential

from keras import layers
model = Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10))
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.15)
history = model.fit(x_train, y_train, epochs=10, 

                    validation_data=(X_val, Y_val))
predictions = model.predict_classes(x_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)
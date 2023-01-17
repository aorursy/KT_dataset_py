# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow import keras

from keras.utils.np_utils import to_categorical 

print(tf.__version__)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

sample=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
train.head()
train.isnull().sum()
X_train=train.drop(['label'], axis=1)/255

X_test=test/255

X_train=X_train.values.reshape(-1, 28,28,1)

X_test=X_test.values.reshape(-1, 28,28,1)
X_train[1].shape
im1=train.drop(['label'], axis=1).loc[10].values.reshape(28,28)

plt.imshow(im1)
Y_train=to_categorical(train['label'], num_classes=10)

Y_train
plt.imshow(X_train[1000,:,:,0])
model =keras.Sequential([

    keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)),

    keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu'),

    keras.layers.MaxPool2D(pool_size=(2,2)),

    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'),

    keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'),

    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),

    keras.layers.Dense(256, activation = "relu"),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(10, activation = "softmax")    

])
model.compile(optimizer='adam',

              loss="categorical_crossentropy",

              metrics=['accuracy'])
Xtrain.shape
from sklearn.model_selection import train_test_split

Xtrain, X_val, Ytrain, Y_val = train_test_split(X_train, Y_train, 

                                                  test_size = 0.1, random_state=2)
model.fit(Xtrain, Ytrain, epochs=5, validation_data=(X_val, Y_val), verbose=1)
results=model.predict(X_test)

results=results.argmax(axis=1)

sub=pd.DataFrame({'ImageId':sample['ImageId'], 'Label': results})

sub.to_csv('sub_mnist.csv', index=False)
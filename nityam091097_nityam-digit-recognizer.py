%matplotlib inline

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
train= pd.read_csv("../input/digit-recognizer/train.csv")

test=pd.read_csv("../input/digit-recognizer/test.csv")

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import matplotlib.image as mp

import seaborn as sns

import matplotlib.pyplot as plt
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 
X_train=X_train/255.0

test= test/255.0
X_train = np.array(X_train).reshape(-1, 28, 28, 1)

test = np.array(test).reshape(-1, 28, 28, 1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=4)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices(test).batch(32)
input_shape = (28,28,1)

output_size = 10



input_layer = tf.keras.Input(shape=input_shape)

conv2d = tf.keras.layers.Conv2D(32, 3)(input_layer)

maxpool = tf.keras.layers.MaxPool2D(3)(conv2d)

conv2d_1 = tf.keras.layers.Conv2D(64, 3)(maxpool)

maxpool_1 = tf.keras.layers.GlobalMaxPool2D()(conv2d_1)



output_layer = tf.keras.layers.Dense(output_size, activation='softmax')(maxpool_1)



model = tf.keras.Model(input_layer, output_layer)
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
X_train[0].shape
model.fit(train_dataset, epochs=10, batch_size=32, validation_data=val_dataset)
x=[np.argmax(x) for x in model.predict(test_dataset.take(1))]
plt.imshow(test[2].reshape(28,28))
df["ImageId"]=df.index

df=df.reset_index(drop=True)

df["Label"]=pd.DataFrame(x)

df.to_csv("submission1.csv")
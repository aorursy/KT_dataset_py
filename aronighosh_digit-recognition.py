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
import csv

import numpy as np

import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.keras import layers

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from os import getcwd

from sklearn.model_selection import train_test_split
train = pd.read_csv ("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv ("/kaggle/input/digit-recognizer/test.csv")

train.shape, test.shape
train.head()
test.head()
train.isna().sum()
Y_train=train['label']

X_train=train.drop(['label'],axis=1)
X_train.head()
X_train=X_train/255.0

test=test/255.0
X_train.shape,test.shape
X_train=X_train.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)

X_train.shape,test.shape
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(Y_train, num_classes = 10)

Y_train = Y_train.astype("int8")
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.22, random_state=42)


model = tf.keras.models.Sequential([



    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

    ])



# Compile Model. 

model.compile(loss = 'categorical_crossentropy',

              optimizer='rmsprop', metrics=['acc'])

train_datagen = ImageDataGenerator(

    rescale=1/255,

    zoom_range=0.2,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    rotation_range=40,

    horizontal_flip=True,

    fill_mode='nearest'

    )



validation_datagen=ImageDataGenerator(rescale=1/255)



train_datagen.fit(x_train)

validation_datagen.fit(x_val)

model.fit(x_train, y_train,batch_size=50, epochs=10)
test_predictions = model.predict(test)





results = np.argmax(test_predictions,axis = 1) 



results = pd.Series(results,name="Label")





submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("digit_recognition.csv",index=False)
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
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
tf.disable_eager_execution()
mnist=keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test)=mnist
# x_train=x_train.reshape(-1,28*28)
# x_test=x_test.reshape(-1,28*28)
# y_train = tf.keras.utils.to_categorical(y_train, 10)
# y_test = tf.keras.utils.to_categorical(y_test, 10)

model=keras.Sequential([#keras.layers.Input(shape=(None,28,28,1)),
                        keras.layers.Conv2D(32,(5,5),padding='same',strides=1,input_shape=(28,28,1)),
                        keras.layers.MaxPooling2D(2),
                        keras.layers.BatchNormalization(),
                        keras.layers.Conv2D(64,(5,5),padding='same',strides=1),
                        keras.layers.MaxPooling2D(2),
                        keras.layers.Dropout(0.2),
                        keras.layers.Flatten(),
                        keras.layers.Dense(1024,activation='relu'),
                        keras.layers.Dense(10,activation='softmax'),
                        
                       ])

model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from tensorflow.keras import backend as K
print(K.image_data_format())
x_np=x_train.reshape(-1,28,28,1)
model.fit(x_np,y_train,epochs=5)
x_re_test=x_test.reshape(-1,28,28,1)
score=model.evaluate(x_re_test,y_test)
score

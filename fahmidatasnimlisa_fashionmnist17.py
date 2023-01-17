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
#!pip install --upgrade pip
#!pip install tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf

print(tf.__version__)
mnist= tf.keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels)=mnist.load_data()
training_images=training_images.reshape(60000,28,28,1)

training_images=training_images/255.0

test_images=test_images.reshape(10000,28,28,1)

test_images=test_images/255.0
model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),

                                  tf.keras.layers.MaxPooling2D(2,2),

                                  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

                                  tf.keras.layers.MaxPooling2D(2,2),

                                  tf.keras.layers.Flatten(),

                                  tf.keras.layers.Dense(1024,activation=tf.nn.relu),

                                  tf.keras.layers.Dense(512,activation=tf.nn.relu),

                                  tf.keras.layers.Dense(10,activation=tf.nn.softmax)])

model.summary
model.compile(optimizer=tf.optimizers.Adam(),

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])

#model.summary()

model.fit(training_images,training_labels,epochs=15)
model.evaluate(test_images,test_labels)
classification=model.predict(test_images)

print(classification[4])

print(test_labels[4])
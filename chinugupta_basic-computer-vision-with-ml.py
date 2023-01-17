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



mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
import matplotlib.pyplot as plt

plt.imshow(training_images[0])

print(training_labels[0])

print(training_images[0])
training_images  = training_images / 255.0

test_images = test_images / 255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 

                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 

                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)]) # Here 10 represents the no. of classes we have in our dataset.

                                                                                          # It is the corresponding no. given to each object in the dataset

                                                                                          # There are two reasons for doing this: Firstly, computer is good with numbers and the second is biasing

                                                                                          # If i would label it as ankle boot then it would be biased and won't learn anything new.

model.compile(optimizer = tf.keras.optimizers.Adam(),

              loss = 'sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
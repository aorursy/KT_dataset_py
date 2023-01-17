# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
import matplotlib.pyplot as plt
#how it looks like

plt.imshow(x_train[2])
#normalization is needed in some cases, so it is better to do it now too

x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test = tf.keras.utils.normalize(x_test, axis=1)
plt.imshow(x_train[2]) #values are between 0 and 1 now
classifier = tf.keras.models.Sequential() #creating the model
classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))#adding layer 1
classifier.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))#adding layer 2
classifier.add(tf.keras.layers.Dense(10, activation=tf.nn.sigmoid))#output layer
classifier.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy']) #compile with basic settings
#now the model is ready

classifier.fit(x_train,y_train,epochs=3)
val_loss, val_acc = classifier.evaluate(x_test, y_test)  # evaluate the out of sample data with model

print(val_loss)  # model's loss (error)

print(val_acc)  # model's accuracy
prediction=classifier.predict(x_test)
prediction[0]
np.argmax(prediction[5])
plt.imshow(x_test[5])
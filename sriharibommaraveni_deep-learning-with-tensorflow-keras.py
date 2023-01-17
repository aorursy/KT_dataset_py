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

import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis = 1)

x_test = tf.keras.utils.normalize(x_test,axis = 1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))

model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))

model.add(tf.keras.layers.Dense(10,activation= tf.nn.softmax))

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy",metrics =["accuracy"])

model.fit(x_train,y_train,epochs = 3)



val_loss,val_acc = model.evaluate(x_test,y_test)

print(val_loss,val_acc)
plt.imshow(x_train[0])

plt.show()

#if you need to get this into black & white.

plt.imshow(x_train[0],cmap = plt.cm.binary)

plt.show()
model.save("epic_num.reader.model")

new_model = tf.keras.models.load_model("epic_num.reader.model")
predictions = new_model.predict([x_test])

print(predictions)
import numpy as np

print(np.argmax(predictions[130]))
plt.imshow(x_test[130])
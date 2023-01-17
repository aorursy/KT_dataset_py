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

tf.__version__
mnist = tf.keras.datasets.mnist # 28*28 images of hand-written digits 0-9



(x_train, y_train),(x_test, y_test) = mnist.load_data()
print(x_train[0])
import matplotlib.pyplot as plt



plt.imshow(x_train[0],cmap=plt.cm.binary) # this will show the number

plt.show()
print(y_train[0])
x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test = tf.keras.utils.normalize(x_test, axis=1)



print(x_train[0])
plt.imshow(x_train[0],cmap=plt.cm.binary)

plt.show()
model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Flatten()) # our input layer

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=3)
val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_loss)

print(val_acc)
model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')

predictions = new_model.predict(x_test)

print(predictions)
import numpy as np



print(np.argmax(predictions[0]))
plt.imshow(x_test[0],cmap=plt.cm.binary)

plt.show()
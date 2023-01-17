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

import matplotlib.pyplot as plt

%matplotlib inline

tf.__version__
mnist=tf.keras.datasets.mnist   #28*28 handwritten digit 0-9 data

(X_train, y_train), (X_test, y_test)=mnist.load_data()

X_train=tf.keras.utils.normalize(X_train, axis=1)

X_test=tf.keras.utils.normalize(X_test, axis=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



model.fit(X_train, y_train, epochs=10)
val_loss, val_acc=model.evaluate(X_test, y_test)

print(val_loss, val_acc)
model.save('mymodel_digit_recognizer')

new_model=tf.keras.models.load_model('mymodel_digit_recognizer')

predictions=new_model.predict(X_test)
print(np.argmax(predictions[10]))
plt.imshow(X_test[10])

plt.show
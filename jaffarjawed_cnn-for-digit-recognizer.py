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

from tensorflow.keras.layers import Flatten, Dense

from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import RMSprop
print(tf.__version__)
x = pd.read_csv('../input/digit-recognizer/train.csv')

y = x.label

x.drop(['label'], inplace = True, axis = 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

x_data = pd.read_csv('../input/digit-recognizer/test.csv')

x_data = x_data.values.reshape(-1, 28, 28, 1)
X_train = X_train.values.reshape(-1, 28, 28, 1)

X_test = X_test.values.reshape(-1, 28, 28, 1)
X_train, X_test = X_train / 255.0, X_test / 255.0
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2 ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation = 'relu'),

    tf.keras.layers.Dense(10, activation = 'softmax')

])

model.summary()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])

history = model.fit(X_train, y_train, epochs = 80, verbose = 1, validation_data = (X_test, y_test))
%matplotlib inline

import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', 'Training Accuracy')

plt.plot(epochs, val_acc, 'b', 'Validation Accuracy')

plt.title("Training and Validation Accuracy")

plt.figure()



plt.plot(epochs, loss, 'r', "Training_loss")

plt.plot(epochs, val_loss, 'b', "Validation_loss")

plt.figure()
predict = model.predict(x_data)

predict = np.argmax(predict, axis = 1)

predict = pd.Series(predict, name = 'Label')
predict
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predict],axis = 1)



submission.to_csv("submission9.csv",index=False)
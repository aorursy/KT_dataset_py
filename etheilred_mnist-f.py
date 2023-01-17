# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf
df = pd.read_csv('../input/k49_classmap.csv')

char_set = pd.read_csv('../input/kmnist_classmap.csv')
char_set
x_train_file, x_test_file = np.load('../input/k49-train-imgs.npz'), np.load('../input/k49-test-imgs.npz')
x_test = x_test_file['arr_0']

x_train = x_train_file['arr_0']
np.true_divide(x_test, 255)

np.true_divide(x_train, 255)
y_train_file, y_test_file = np.load('../input/k49-train-labels.npz'), np.load('../input/k49-test-labels.npz')
y_test = y_test_file['arr_0']

y_train = y_train_file['arr_0']
x_train = x_train / 255.0

x_test = x_test / 255.0
y_train.shape
model2 = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(28,28)),

    tf.keras.layers.Dense(800, activation='relu'),

    tf.keras.layers.Dense(198, activation='relu'),

    tf.keras.layers.Dense(49, activation=tf.nn.softmax)

])
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=10)
res = model2.evaluate(x_test, y_test, verbose=1)

res[1]

a.append(res[1])
a
X_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

X_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
Y_train = tf.keras.utils.to_categorical(y_train, 49)

Y_test = tf.keras.utils.to_categorical(y_test, 49)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(75, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)),

    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(75, kernel_size=(5,5), activation='relu'),

    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(500, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(49, activation='softmax')

])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size = 200, epochs=10, validation_split=0.2, verbose=1)
res = model.evaluate(X_test, Y_test, verbose=1)

b.append(res[1])

res[1]
i = 1

while (b[-1]>b[-2])and (abs(b[-1]-b[-2])>0.001):

    model.fit(X_train, Y_train, batch_size = 200, epochs=10, validation_split=0.2, verbose=1)

    res = model.evaluate(X_test, Y_test, verbose=1)

    b.append(res[1])

    print('iteration '+str(i)+': '+str(b[-1]))
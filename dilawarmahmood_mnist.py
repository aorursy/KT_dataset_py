import tensorflow as tf

from tensorflow import keras



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
test
train_x = np.reshape(train[train.columns[1:]].values, (-1, 28, 28))

train_y = train['label']



test_x = np.reshape(test[test.columns[:]].values, (-1, 28, 28))
plt.figure()

plt.imshow(train_x[0])

plt.grid(False)

plt.colorbar()

plt.show()
train_x = train_x / 255.0

test_x = test_x / 255.0
plt.figure()

plt.imshow(train_x[0], cmap=plt.cm.binary)

plt.grid(False)

plt.xticks([])

plt.yticks([])

plt.xlabel(train_y[0])

plt.show()
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28,28)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10)

])
model.compile(optimizer='adam',

              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=10)
prob_model = keras.Sequential([

    model,

    keras.layers.Softmax()

])
preds = prob_model.predict(test_x)
plt.figure()

plt.imshow(test_x[69], cmap=plt.cm.binary)

plt.grid(False)

plt.xticks([])

plt.yticks([])

plt.xlabel(np.argmax(preds[69]))

plt.show()
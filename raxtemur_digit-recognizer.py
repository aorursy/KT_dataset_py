import tensorflow as tf

import pandas as pd

import numpy as np



model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,

                           input_shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation=tf.nn.relu),

    tf.keras.layers.Dropout(0.50),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
Train = pd.read_csv("../input/train.csv");

x_test = pd.read_csv("../input/test.csv");
y_train = Train.iloc[:, 0]

x_train = Train.iloc[:, 1:] / 255.0
x_train = np.reshape(x_train.values, (-1, 28, 28, 1))

y_train = y_train.values

print(x_train.shape, y_train.shape)
model.fit(x_train, y_train, epochs=15)
test = pd.read_csv("../input/test.csv")

x_test = np.reshape(x_test.values, (-1, 28, 28, 1));
y_test = model.predict(x_test)

y_test

predictions = np.zeros(28000, dtype=int)

predictions
for i in range(28000):

    for j in range(10):

        if y_test[i, j] == 1:

            predictions[i] = j
predictions = pd.DataFrame({'Label' : predictions})

predictions.index+=1;

predictions.to_csv('predictions.csv', index_label = 'ImageId')
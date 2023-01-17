!pip install tensorflow-gpu==2.0.0.alpha0
import tensorflow as tf

import numpy as np

import pandas as pd
training = pd.read_csv('../input/digit-recognizer/train.csv')
x_train, y_train = training.iloc[:, 1:], training.iloc[:, 0:1]
x_train = x_train / 255
x_train = x_train.values.reshape(-1, 28, 28, 1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding="same", activation="relu", input_shape=[28, 28, 1]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=15)
test = pd.read_csv('../input/digit-recognizer/test.csv')
test = test / 255
test = test.values.reshape(-1, 28, 28, 1)
predictions = model.predict(test)
export = pd.DataFrame([np.argmax(prediction) for prediction in predictions])
export.index += 1 
export = export.reset_index()
export.columns = ['ImageId', 'Label']
export.to_csv('export.csv', index=False)
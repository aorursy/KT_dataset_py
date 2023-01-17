import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
x = train_data.drop('label', axis=1).values.astype(np.float32)
x = x / 255.0
x = x.reshape(-1, 28, 28, 1)

y = train_data.label.values
y = to_categorical(y)

print('Train Images Shape:', x.shape)
print('Train Labels Shape:', y.shape)
test_data = test_data.values.astype(np.float32)
test_data = test_data / 255.0
test_data = test_data.reshape(-1, 28, 28, 1)
print('Test Images Shape:', test_data.shape)
train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(20, (5, 5), padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Conv2D(50, (5, 5), padding='same'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)
history = model.fit(
    train_x,
    train_y,
    epochs=5,
    validation_data=(val_x, val_y),
)
predictions = model.predict(test_data)
predictions = np.argmax(predictions, axis=1)
submission = pd.concat([pd.Series(range(1,28001), name="ImageId"), pd.Series(predictions, name='Label')], axis=1)
submission.head()
submission.to_csv('submission.csv', index=False)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.image as mimg
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
train_df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
test_df = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
print(train_df.shape)
train_df.head()
print(test_df.shape)
test_df.head()
X_train = train_df.drop(['label'], axis=1)
y_train = train_df['label']
X_test = test_df.drop(['label'], axis=1)
y_test = test_df['label']
X_train = np.reshape(X_train.values, (60000, 28, 28))/255
X_test = np.reshape(X_test.values, (10000, 28, 28))/255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
X_train[0]
X_train = np.reshape(X_train, (60000, 28, 28, 1))
X_test = np.reshape(X_test, (10000, 28, 28, 1))
# Build the Model
input_shape = (28, 28, 1)

model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation = 'relu', input_shape = input_shape),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, 3, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, 5, strides=2, padding='same', activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    
    keras.layers.Conv2D(64, 3, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, 3, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, 5, strides=2, padding='same', activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
        
    keras.layers.Conv2D(128, 4, activation = 'relu'),
    keras.layers.BatchNormalization(),
    
    keras.layers.Flatten(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation = 'softmax')
])

model.summary()
epochs = 15
batch_size = 64
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, 
                    validation_split=0.1)
history = model.fit(X_train, y_train, epochs = 10, batch_size = batch_size, 
                    validation_split=0.1)
history = model.fit(X_train, y_train, epochs = 10, batch_size = batch_size, 
                    validation_split=0.1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
model.evaluate(X_test, y_test)
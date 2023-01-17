import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np



%matplotlib inline
# load data

from tensorflow.keras.datasets import cifar10

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Loading the dataset

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalization

X_train = X_train / 255.0
X_train.shape
X_test = X_test / 255.0
plt.imshow(X_test[99])

# plot some first images 0-99



plt.figure()

fig, axes = plt.subplots(10, 10)



#ravel (np.) returns contiguous flattened array

for i, ax in enumerate(axes.ravel()):

    image = ax.imshow(X_test[i])



plt.show()
# defining the model

model = tf.keras.models.Sequential()



# first CNN layer

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))



# second CNN layer and max pool layer & dropout

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))

model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

model.add(tf.keras.layers.Dropout(0.3))



# third CNN Layer

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))



# fourth CNN Layer and max pool layer & dropout

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

model.add(tf.keras.layers.Dropout(0.3))



# flatten

model.add(tf.keras.layers.Flatten())



# first Dense layer

model.add(tf.keras.layers.Dense(units=128, activation='relu'))



# output layer second Dense layer

model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.summary()



# model compile (for classification)

model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["sparse_categorical_accuracy"])
# model train

model.fit(X_train, y_train, epochs=15)
# model evaluation

test_loss, test_accuracy = model.evaluate(X_test, y_test)
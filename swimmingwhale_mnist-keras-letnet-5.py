import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:,:,:,np.newaxis]
y_train = y_train[:,np.newaxis]
x_test = x_test[:,:,:,np.newaxis]
y_test = y_test[:,np.newaxis]
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
fig, ax = plt.subplots(
    nrows=5,
    ncols=5,
    sharex='all',
    sharey='all', )

ax = ax.flatten()
for i in range(25):
    img = x_train[i][:,:,0]
    ax[i].imshow(img,cmap='gray')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
def lenet_model(img_shape,n_classes,l2_reg, weights=None):
    # Initialize model
    model = Sequential()
    # 1 layer
    model.add(Conv2D(20, (5, 5), padding="same",input_shape=img_shape, kernel_regularizer=l2(l2_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # 2 layer
    model.add(Conv2D(50, (5, 5), padding="same",kernel_regularizer=l2(l2_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(500, kernel_regularizer=l2(l2_reg)))
    model.add(Activation("relu"))
    # Softmax
    model.add(Dense(n_classes, kernel_regularizer=l2(l2_reg)))
    model.add(Activation("softmax"))
    model.summary()
    if weights is not None:
        model.load_weights(weights)
    return model
model = lenet_model((28, 28, 1), 10, 0.01)
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2, batch_size=32)
model.evaluate(x_test, y_test, batch_size=128)
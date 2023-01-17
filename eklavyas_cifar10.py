import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os



from keras.datasets import cifar10
from sklearn.model_selection import train_test_split

from sklearn import metrics
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam

from keras.regularizers import l2

from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
# Normalise data

x_train = (x_train - x_train.mean()) / x_train.std()

x_val = (x_val - x_val.mean()) / x_val.std()

x_test = (x_test - x_test.mean()) / x_test.std()
# Other variables

n_classes = 10

height = width = x_train.shape[1]

n_train_samples = x_train.shape[0]

n_channels = x_train.shape[3]

n_features = 32
# Reshape

y_train = y_train.flatten()

y_val = y_val.flatten()

y_test = y_test.flatten()
# One-hot encode labels

y_train = np.eye(np.max(y_train)+1)[y_train]

y_val  = np.eye(np.max(y_val)+1)[y_val]

y_test  = np.eye(np.max(y_test)+1)[y_test]
# Build model

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))

model.add(Activation('relu'))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(n_classes))

model.add(Activation('softmax'))
# Compile model

model.compile(loss=categorical_crossentropy,

              optimizer=Adam(learning_rate=1e-4),

              metrics=['accuracy'])
# Fit model

history = model.fit(x_train, y_train, epochs=75, verbose=1, validation_data=(x_val, y_val), shuffle=True)
# Score model

score = model.evaluate(x_test, y_test, verbose=0)

print('Test accuracy:', np.round(score[1], 2))

print('Test loss:', np.round(score[0], 2))
# Plot accuracy

plt.figure(dpi=100)

plt.plot(model.history.history['accuracy'], label='Train accuracy')

plt.plot(model.history.history['val_accuracy'], label='Validation accuracy')

plt.legend()

plt.show()
# Plot loss

plt.figure(dpi=100)

plt.plot(model.history.history['loss'], label='Train loss')

plt.plot(model.history.history['val_loss'], label='Validation loss')

plt.legend()

plt.show()
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import keras
from collections import Counter
from sklearn.model_selection import train_test_split

from sklearn import metrics
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam

from keras.regularizers import l2

from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")

test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")
# Get variables

x_train = train.loc[:, train.columns != 'label'].values

y_train = train.label.values



x_test = test.loc[:, train.columns != 'label'].values

y_test = test.label.values
# Get validation data

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
# Normalise training data

x_train = (x_train - x_train.mean()) / x_train.std()

x_val = (x_val - x_val.mean()) / x_val.std()

x_test = (x_test - x_test.mean()) / x_test.std()
# Other variables

n_classes = len(train.label.unique())

height = width = int(np.sqrt(x_train.shape[1]))

n_train_samples = x_train.shape[0]

n_features = 32
# Reshape input data

x_train = x_train.reshape(x_train.shape[0], height, width, 1)

x_val  = x_val.reshape(x_val.shape[0], height, width, 1)

x_test  = x_test.reshape(x_test.shape[0], height, width, 1)
# One-hot encode labels

y_train = np.eye(np.max(y_train)+1)[y_train]

y_val  = np.eye(np.max(y_val)+1)[y_val]

y_test  = np.eye(np.max(y_test)+1)[y_test]
model = keras.Sequential()



model.add(Conv2D(filters=n_features, kernel_size=(3,3), activation='relu', input_shape=(height, width, 1)))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(filters=2*n_features, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(filters=2*2*n_features, kernel_size=(3,3), activation='relu'))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dropout(0.5))



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(n_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adam(learning_rate=1e-4),

              metrics=['accuracy'])
#model.summary()
history = model.fit(x_train, y_train, epochs=75, verbose=2, validation_data=(x_val, y_val))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', np.round(score[0], 2))

print('Test accuracy:', np.round(score[1], 2))
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
# Test evaluation

probs = model.predict(x_test, verbose=0)
# Get accuracy

np.mean(y_hat==y_real)

y_hat = np.argmax(probs, axis=1)

y_real = np.argmax(y_train, axis=1)
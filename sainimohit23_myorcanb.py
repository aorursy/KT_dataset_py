import numpy as np

import keras

import os

import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

import time

%matplotlib inline
X_train = np.append(np.load("../input/images.npy"), np.load("../input/nimages.npy"), axis=0)

Y_train = np.append(np.load("../input/labels.npy"), np.load("../input/nlabels.npy"), axis=0)

for i in range(X_train.shape[0]):

    X_train[i] = (X_train[i]/255)

    X_train[i] = X_train[i] -  0.5

    X_train[i] = 2*X_train[i]
num=11

plt.imshow(X_train[num])

print(Y_train[num])
inp = keras.layers.Input(shape=(320, 480, 3))

X = keras.layers.Conv2D(32, (2, 2), activation='relu', padding='same')(inp)

X = keras.layers.Conv2D(48, (2, 2), activation='relu', padding='same')(X)

X = keras.layers.MaxPool2D()(X)

X = keras.layers.Dropout(0.5)(X)

X = keras.layers.Flatten()(X)

X = keras.layers.Dense(128, activation='relu')(X)

X = keras.layers.Dropout(0.3)(X)

X = keras.layers.Dense(64, activation='relu')(X)

X = keras.layers.Dropout(0.4)(X)

X = keras.layers.Dense(8, activation='softmax')(X)



model = keras.models.Model(inp, X)

# c = ModelCheckpoint("../input/model.h5", monitor='train_loss')

opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-3, decay=1e-6, amsgrad=False)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x=X_train, y= Y_train, batch_size=8, epochs=100, shuffle=True)
model.save("./model.h5")
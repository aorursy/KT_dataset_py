import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras import optimizers
X = np.load("../input/Sign-language-digits-dataset/X.npy")

Y = np.load("../input/Sign-language-digits-dataset/Y.npy")
X.shape
Y.shape
plt.imshow(X[0])
print(Y[0])
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=8)

Xtrain = Xtrain[:,:,:,np.newaxis]

Xtest = Xtest[:,:,:,np.newaxis]
model = Sequential()

model.add(Conv2D(input_shape=(64, 64, 1), filters=64, kernel_size=(4,4)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=4))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',

             optimizer=optimizers.Adadelta(),

             metrics=['accuracy'])

model.fit(Xtrain, Ytrain, batch_size=32, epochs=10)

score = model.evaluate(Xtest, Ytest)
print("Loss is {0:.2f}\nAccuracy is {1:.2f}%".format(score[0], score[1]*100))
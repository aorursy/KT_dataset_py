import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Dropout

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dense

from keras.optimizers import SGD

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from keras.layers.normalization import BatchNormalization

import os
X = np.load('../input/Sign-language-digits-dataset/X.npy')

y = np.load('../input/Sign-language-digits-dataset/Y.npy')

print(X.shape)

print(y.shape)

X = X.reshape(-1, X.shape[1], X.shape[2], 1)
print(min(X.flatten()))

print(max(X.flatten()))
y_labels = np.where(y == 1)[1]

bars = np.asarray(range(0,10,1))

n, bins, patches = plt.hist(x=y_labels, bins=len(bars), color='#0504aa',

                            alpha=0.7, rwidth=0.85)

bins += 0.5

plt.xticks(bins,bars)

plt.xlabel('Digit', fontsize=14)

plt.ylabel('Frequency', fontsize=14)
class LeNet:



    def build(width, height, depth, classes):

    

        model = Sequential()

        inputShape = (height, width, depth)



        model.add(Conv2D(20, 5, padding="same",

                         input_shape=inputShape))

        model.add(Activation("relu"))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



        model.add(Conv2D(50, 5, padding="same"))

        model.add(Activation("relu"))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



        model.add(Flatten())

        model.add(Dense(500))

        model.add(Activation("relu"))



        model.add(Dense(classes))

        model.add(Activation("softmax"))



        return model

class MiniVGGNet:

    

    def build(width, height, depth, classes):

        model = Sequential()

        inputShape = (height, width, depth)

        

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(32, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=-1))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))

        

        model.add(Conv2D(64, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(64, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=-1))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))

        

        model.add(Flatten())

        model.add(Dense(500))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))

        

        model.add(Dense(classes))

        model.add(Activation("softmax"))

        

        return model
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.05, random_state=42)

print("[INFO] compiling model...")

opt = SGD(lr=0.01)

model1 = LeNet.build(width=64, height=64, depth=1, classes=10)

model1.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print("[INFO] training network...")

H = model1.fit(trainX, trainY, validation_split=0.15, batch_size=128, epochs=50, verbose=1)
print("[INFO] evaluating network...")

predictions = model1.predict(testX, batch_size=128)

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in np.unique(y_labels)]))
plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, 50), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, 50), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, 50), H.history["acc"], label="train_acc")

plt.plot(np.arange(0, 50), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.show()
model2 = MiniVGGNet.build(width=64, height=64, depth=1, classes=10)

model2.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H1 = model2.fit(trainX, trainY, validation_split=0.15, batch_size=128, epochs=50, verbose=1)
print("[INFO] evaluating network...")

predictions = model2.predict(testX, batch_size=128)

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in np.unique(y_labels)]))
plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, 50), H1.history["loss"], label="train_loss")

plt.plot(np.arange(0, 50), H1.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, 50), H1.history["acc"], label="train_acc")

plt.plot(np.arange(0, 50), H1.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.show()
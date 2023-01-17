from keras.datasets import mnist

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras import backend as K

import keras



batch_size = 32

epochs = 10



(x_train, y_train), (x_test, y_test) = mnist.load_data()



img_rows = x_train[0].shape[0]

img_cols = x_train[1].shape[0]



x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)



input_shape = (img_rows, img_cols, 1)



x_train = x_train.astype("float32")

x_test = x_test.astype("float32")



x_train /= 255

x_test /= 255



y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)



num_classes = y_test.shape[1]

num_pixels = x_train.shape[1] * x_train.shape[2]



model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation="softmax"))



model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adadelta(), metrics=["accuracy"])

print(model.summary())



history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=0)

print("Test Loss:", scores[0])

print("Test Accuracy:", scores[1])
import matplotlib.pyplot as plt



history_dict = history.history



loss_values = history_dict["loss"]

val_loss_values = history_dict["val_loss"]

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_loss_values, label="Validation/Test Loss")

line2 = plt.plot(epochs, loss_values, label="Training Loss")

plt.setp(line1, linewidth=2.0, marker="+", markersize=10.0)

plt.setp(line2, linewidth=2.0, marker="4", markersize=10.0)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.grid(True)

plt.legend()

plt.show()
acc_values = history_dict["accuracy"]

val_acc_values = history_dict["val_accuracy"]

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_acc_values, label="Validation/Test Loss")

line2 = plt.plot(epochs, acc_values, label="Training Loss")

plt.setp(line1, linewidth=2.0, marker="+", markersize=10.0)

plt.setp(line2, linewidth=2.0, marker="4", markersize=10.0)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.grid(True)

plt.legend()

plt.show()
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np



y_pred = model.predict_classes(x_test)

print(classification_report(np.argmax(y_test, axis=1), y_pred))

print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))
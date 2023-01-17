import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split

from tensorflow.python.keras.layers.kernelized import RandomFourierFeatures 

import matplotlib.pyplot as plt

import matplotlib as mpl

%matplotlib inline

%reload_ext autoreload

%autoreload 2
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
train.head(2)
y_train = train.label

train.drop(columns=["label"], inplace=True)

y_train
x_train = train.values /255

x_test = test.values /255

x_train = x_train.reshape(-1,28,28,1)

x_test = x_test.reshape(-1,28,28,1)
for index in range(15):

    image = x_train[index]

    label = y_train[index]

    plt.subplot(1, 15, index + 1)

    plt.imshow(image.reshape(28,28), cmap="binary")

    plt.axis("off")

    plt.title(str(label))
x_train = x_train.reshape(-1, 784).astype('float32') / 255

x_test = x_test.reshape(-1, 784).astype('float32') / 255



x_train,x_valid = train_test_split(x_train, test_size=0.3)

y_train,y_valid = train_test_split(y_train, test_size=0.3)

yvalid =y_valid 

y_valid.shape,x_valid.shape
y_train = keras.utils.to_categorical(y_train)

y_valid = keras.utils.to_categorical(y_valid)

y_train.shape,y_valid.shape
keras.backend.clear_session()

model = keras.Sequential()

model.add(keras.layers.Input(shape=(784,)))

frr = RandomFourierFeatures(output_dim=4096,scale=10.,kernel_initializer='gaussian')

model.add(frr)

model.add(keras.layers.Dense(256, activation="relu"))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(units=10))
model.compile(loss=keras.losses.hinge, optimizer=keras.optimizers.RMSprop(lr=1e-4), metrics=[keras.metrics.CategoricalAccuracy(name='acc')])

model.summary()
fit = model.fit(x_train, y_train, epochs=5, validation_split=0.2)


rff_loss = fit.history["loss"]

rff_val_loss = fit.history["val_loss"]

plt.plot(np.arange(len(rff_loss)) + 0.5, rff_loss, "b.-", label="Training loss")

plt.plot(np.arange(len(rff_val_loss)) + 1, rff_val_loss, "r.-", label="Validation loss")

plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

plt.axis([1, 5, 0, .6])

plt.legend(fontsize=14)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.grid(True)

plt.show()
model.evaluate(x_valid, y_valid)
p_valid = model.predict_classes(x_valid)

np.mean(keras.losses.mean_squared_error(yvalid, p_valid))

predictions = model.predict_classes(x_test)

predictions[:10]
keras.backend.clear_session()

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
y_train = train.label

train.drop(columns=["label"], inplace=True)

y_train = keras.utils.to_categorical(y_train)
x_train = train.values

x_test = test.values

x_train = x_train.reshape(-1,28,28,1).astype(np.float64)

x_test = x_test.reshape(-1,28,28,1).astype(np.float64)

x_train = x_train.astype(np.float64)

x_test = x_test.astype(np.float64)

x_train[1].shape
keras.backend.clear_session()

kernel_size = 3

filters=64

model = keras.models.Sequential()



#model.add(keras.Input(shape=(28, 28,1)))



model.add(keras.layers.Conv2D(filters=filters,input_shape=(28, 28,1), kernel_size=kernel_size, activation="relu"))

model.add(keras.layers.MaxPool2D(2))



model.add(keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation="relu"))

model.add(keras.layers.MaxPool2D(2))



model.add(keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation="relu"))



model.add(keras.layers.Flatten())

model.add(keras.layers.Dropout(0.2))



model.add(keras.layers.Dense(10, activation="softmax"))



model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()
plateau_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=.5, min_lr=.00001)

callbacks=[plateau_callback]



fit = model.fit(x_train, y_train, epochs=50, validation_split=0.3,callbacks=[plateau_callback], batch_size=128)
loss = fit.history["loss"]

val_loss = fit.history["val_loss"]

plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")

plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")

plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

plt.axis([1, 5, 0, .5])

plt.legend(fontsize=14)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.grid(True)

plt.show()



acc = fit.history["accuracy"]

val_acc = fit.history["val_accuracy"]

plt.plot(np.arange(len(loss)) + 0.5, acc, "b.-", label="Training acc")

plt.plot(np.arange(len(val_loss)) + 1, val_acc, "r.-", label="Validation acc")

plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

plt.axis([1, 5, 1, .9])

plt.legend(fontsize=14)

plt.xlabel("Epochs")

plt.ylabel("Acc")

plt.grid(True)

plt.show()

predictions = model.predict_classes(x_test)

predictions[:10]
for index in range(10):

    image = x_test[index]

    label = predictions[index]

    plt.subplot(1, 10, index + 1)

    plt.imshow(image.reshape(28,28), cmap="binary")

    plt.axis("off")

    plt.title(str(label))
submission['Label'] = predictions

submission.to_csv("submission_cnn.csv" , index = False)

submission.head()
loss = fit.history["loss"]

plt.plot(np.arange(len(rff_loss)) + 1, rff_loss, "r.-", label="RFF loss")

plt.plot(np.arange(len(loss)) + 1, loss, "g.-", label="CNN loss")

plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

plt.axis([1, 5, 0, .6])

plt.legend(fontsize=14)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.grid(True)

plt.show()
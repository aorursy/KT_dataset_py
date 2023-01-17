from keras import backend as K, metrics

from keras.layers import (

    Input,

    Activation,

    Dense,

    Flatten

)

from keras.layers.normalization import BatchNormalization

from keras.models import Model, Sequential

from keras.optimizers import SGD, Adam

import matplotlib.pyplot as plt

import numpy as np



class Shallow1D:

    def __call__(self):

        inputs = Input(shape=(1,))

        # 4 hidden nodes and 2000 iterations work too.

        x = Dense(16, activation="tanh", kernel_initializer='he_normal')(inputs)

        y = Dense(1, activation="linear")(x)

        model = Model(inputs=inputs, outputs=y)

        return model

    

class Deep1D:

    def __call__(self):

        inputs = Input(shape=(1,))

        x = Dense(4, activation="tanh", kernel_initializer='he_normal')(inputs)

        x = Dense(4, activation="tanh", kernel_initializer='he_normal')(x)

        x = Dense(4, activation="tanh", kernel_initializer='he_normal')(x)

        x = Dense(4, activation="tanh", kernel_initializer='he_normal')(x)

        y = Dense(1, activation="linear")(x)

        model = Model(inputs=inputs, outputs=y)

        return model



def plot_result(X, Y, loss, val_loss, fun):

    epochs = range(1, len(loss) + 1)

    fig = plt.figure(figsize=(14,6))

    ax1 = fig.add_subplot(121)

    ax2 = fig.add_subplot(122)

    ax1.plot(epochs, loss, "r--", label="Training")

    ax1.plot(epochs, val_loss, "r", label="Validation")

    ax1.set_title("Training and validation loss")

    ax1.legend()

    Xref = np.linspace(0, 2 * np.pi, 256)

    ax2.plot(Xref, fun(Xref), color="r", label="sin(x)")

    ax2.scatter(X, Y, s=6, label="approximation")

    ax1.set_xlabel("Epochs")

    ax1.set_ylabel("Error")

    ax1.set_title("Training error")

    ax2.set_title("Decision boundary")

    ax2.set_xlabel("x")

    ax2.set_ylabel("y")

    ax2.legend()

    plt.show()
def func1d(x):

    return np.sin(x)



n_training_examples = 1024

X0 = np.random.rand(n_training_examples, 1) * 2 * np.pi

T = np.reshape(func1d(X0), (n_training_examples, 1))



# Train model.

n_epochs = 200

model = Shallow1D()()

model.compile(optimizer=Adam(), loss="mse")

model.count_params()

model.summary()

history = model.fit(X0, T, validation_split=0.2, epochs=n_epochs, verbose=0)



# Test model.

n_test_examples = 100

X = np.linspace(0, 2 * np.pi, n_test_examples)

Y = model.predict(X)



loss = history.history["loss"]

val_loss = history.history["val_loss"]

print("Best loss: {:f} (trn), {:f} (val)".format(np.min(loss), np.min(val_loss)))

plot_result(X, Y, loss, val_loss, func1d)
n_training_examples = 1024

X0 = np.random.rand(n_training_examples, 1) * 2 * np.pi

T = np.reshape(func1d(X0), (n_training_examples, 1))



# Train model.

n_epochs = 200

model = Deep1D()()

model.compile(optimizer=Adam(), loss="mse")

model.count_params()

model.summary()

history = model.fit(X0, T, validation_split=0.2, epochs=n_epochs, verbose=0)



# Test model.

n_test_examples = 100

X = np.linspace(0, 2 * np.pi, n_test_examples)

Y = model.predict(X)



loss = history.history["loss"]

val_loss = history.history["val_loss"]

print("Best loss: {:f} (trn), {:f} (val)".format(np.min(loss), np.min(val_loss)))

plot_result(X, Y, loss, val_loss, func1d)
from keras import backend as K, metrics

from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D

from keras.models import Model

from keras.optimizers import Adam

from keras.utils import to_categorical

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



class FashionClassifier:

    def __call__(self):

        inputs = Input(shape=(28, 28, 1))

        x = Flatten()(inputs)

        x = Dense(128, kernel_initializer='he_normal', use_bias=False)(x)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        category = Dense(10, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=category)

        return model



def fashion_mnist():

    data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

    data_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')



    X_train = np.array(data_train.iloc[:, 1:])

    X_test = np.array(data_test.iloc[:, 1:])

    y_train = to_categorical(np.array(data_train.iloc[:, 0]))

    y_test = to_categorical(np.array(data_test.iloc[:, 0]))



    img_rows, img_cols = 28, 28

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)



    X_train = X_train.astype("float32")

    X_test = X_test.astype("float32")

    X_train /= 255

    X_test /= 255

    return (X_train, y_train), (X_test, y_test)



(train_images, train_labels), (test_images, test_labels) = fashion_mnist()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape, train_labels.shape)



plt.figure(figsize=(10, 10))

for i in range(25):

    plt.subplot(5, 5, i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i, :, :, 0], cmap="binary")

    plt.xlabel(class_names[np.argmax(train_labels[i])])

plt.show()
model = FashionClassifier()()

model.compile(optimizer=Adam(), 

              loss='categorical_crossentropy',

              metrics=['accuracy'])

model.count_params()

model.summary()

history = model.fit(train_images, train_labels, validation_split=0.15, epochs=10)



acc = history.history["acc"]

val_acc = history.history["val_acc"]

print("Best accuracy: {:f} (trn), {:f} (val)".format(np.max(acc), np.max(val_acc)))



epochs = range(1, len(acc) + 1)

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.plot(epochs, acc, "r--", label="Training")

ax1.plot(epochs, val_acc, "r", label="Validation")

ax1.set_title("Training and validation accuracy")

ax1.legend()

plt.show()
n = 444

prediction = model.predict(test_images[n].reshape((1, 28, 28, 1)))

label = test_labels[n]

for i, p in enumerate(prediction[0]):

    print("{:=5.2f} % - {:s} {:s}".format(p * 100, class_names[i], "<--- correct" if label[i] > 0 else ""))
class FashionCNN:

    def __call__(self):

        inputs = Input(shape=(28, 28, 1))

        x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)

        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Dropout(0.2)(x)

        x = Flatten()(x)

        x = Dense(128, activation="relu")(x)

        category = Dense(10, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=category)

        return model



# Fancier model that you might try instead of FashionCNN below.

class FashionMiniVGGNet:

    """Source: https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/

    Modified to have batch norm before activation (per the original paper but order is debated).

    """

    def __call__(self):

        inputs = Input(shape=(28, 28, 1))

        # first CONV => RELU => CONV => RELU => POOL layer set

        x = Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        x = Conv2D(32, (3, 3), padding="same", use_bias=False)(x)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Dropout(0.25)(x)

 

        # second CONV => RELU => CONV => RELU => POOL layer set

        x = Conv2D(64, (3, 3), padding="same", use_bias=False)(x)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        x = Conv2D(64, (3, 3), padding="same", use_bias=False)(x)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Dropout(0.25)(x)

 

        # first (and only) set of FC => RELU layers

        x = Flatten()(x)

        x = Dense(512, use_bias=False)(x)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        x = Dropout(0.5)(x)

 

        # softmax classifier

        category = Dense(10, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=category)

        return model



model = FashionCNN()()

model.compile(optimizer=Adam(), 

              loss='categorical_crossentropy',

              metrics=['accuracy'])

model.count_params()

model.summary()

history = model.fit(train_images, train_labels, validation_split=0.15, batch_size=64, epochs=10)



acc = history.history["acc"]

val_acc = history.history["val_acc"]

print("Best accuracy: {:f} (trn), {:f} (val)".format(np.max(acc), np.max(val_acc)))



epochs = range(1, len(acc) + 1)

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.plot(epochs, acc, "r--", label="Training")

ax1.plot(epochs, val_acc, "r", label="Validation")

ax1.set_title("Training and validation accuracy")

ax1.legend()

plt.show()
score = model.evaluate(test_images, test_labels, verbose=0)

print("Test loss: {:.4f}".format(score[0]))

print("Test accuracy: {:.4f}".format(score[1]))
from math import ceil

import os



from keras import backend as K, metrics

from keras.layers import (

    Input,

    Activation,

    Dense,

    Flatten,

    Dropout,

    GlobalAveragePooling2D,

)

from keras.layers.convolutional import AveragePooling2D, Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.models import Model, Sequential

from keras.optimizers import SGD, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.regularizers import l2

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np



class ArdisCNN:

    def __call__(self):

        inputs = Input(shape=(48, 48, 3))

        x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)

        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Dropout(0.2)(x)

        

        x = Flatten()(x)

        x = Dense(128, activation="relu")(x)

        category = Dense(10, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=category)

        return model



seed = 42  # For repeatability.

np.random.seed(seed)



batch_size = 64

image_size = (48, 48)

train_dir = "../input/ardis-dataset-3/ardis-dataset-3/"



# Generators for images. Automatically assigns class '3'

# to all images in subdirectory '3'.

# Also uses 20% of the training set for validation during training.

datagen = ImageDataGenerator(

    rescale=1.0 / 255,

    # horizontal_flip=True,

    validation_split=0.2

)

train_generator = datagen.flow_from_directory(

    train_dir,

    target_size=image_size,

    color_mode="rgb",

    interpolation="lanczos",

    batch_size=batch_size,

    class_mode="categorical",

    subset="training",

    seed=seed

)

validation_generator = datagen.flow_from_directory(

    train_dir,

    target_size=image_size,

    color_mode="rgb",

    interpolation="lanczos",  # Fancy but slow interpolation for resizing!

    batch_size=batch_size,

    class_mode="categorical",

    subset="validation",

    seed=seed

)



# Compile model.

model = ArdisCNN()()

model.compile(

    optimizer=Adam(lr=1e-3),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

model.count_params()

model.summary()



# Increase this!

n_epochs = 4



# Train model.

history = model.fit_generator(

    train_generator,

    steps_per_epoch=ceil(train_generator.n / batch_size),

    epochs=n_epochs,

    validation_data=validation_generator,

    validation_steps=ceil(validation_generator.n / batch_size),

)



#

# Plot performance

#

acc = history.history["acc"]

val_acc = history.history["val_acc"]

loss = history.history["loss"]

val_loss = history.history["val_loss"]



print("Best accuracy: {:f} (trn), {:f} (val)".format(np.min(acc), np.min(val_acc)))

print("Best loss: {:f} (trn), {:f} (val)".format(np.min(loss), np.min(val_loss)))



epochs = range(1, len(loss) + 1)

plt.plot(epochs, acc, "b--", label="Training")

plt.plot(epochs, val_acc, "b", label="Validation")

plt.title("Training and validation Accuracy")

plt.legend()

plt.figure()

plt.plot(epochs, loss, "r--", label="Training")

plt.plot(epochs, val_loss, "r", label="Validation")

plt.title("Training and validation loss")

plt.legend()

plt.show()
class_names = list(validation_generator.class_indices.keys())

x, y = next(validation_generator)

prediction = model.predict(x)



plt.figure(figsize=(10, 10))

for i in range(25):

    plt.subplot(5, 5, i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    # plt.imshow(x[i, :, :, 0], cmap="binary")  # For grayscale images.

    plt.imshow(x[i, :, :])  # For color images.

    plt.xlabel("{:s} ({:.0f}%)".format(

        class_names[np.argmax(prediction[i])],

        prediction[i, np.argmax(prediction[i])] * 100

    ))

plt.show()
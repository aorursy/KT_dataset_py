import tensorflow as tf

from tensorflow import keras

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

import sklearn

import matplotlib as mpl

#from livelossplot.tf_keras import PlotLossesCallback

#gpus= tf.config.experimental.list_physical_devices('GPU')

#tf.config.experimental.set_memory_growth(gpus[0], True)
# Load dataset 

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

X_train_full = X_train_full.astype(np.float32) / 255

X_test = X_test.astype(np.float32) / 255

X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]

y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


tf.random.set_seed(42)

np.random.seed(42)



stacked_encoder = keras.models.Sequential([

    keras.layers.Flatten(input_shape=[28, 28]),

    keras.layers.Dense(100, activation="selu"),

    keras.layers.Dense(30, activation="selu"),

])

stacked_decoder = keras.models.Sequential([

    keras.layers.Dense(100, activation="selu", input_shape=[30]),

    keras.layers.Dense(28 * 28, activation="sigmoid"),

    keras.layers.Reshape([28, 28])

])

stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])

stacked_ae.compile(loss="binary_crossentropy",

                   optimizer=keras.optimizers.SGD(lr=1.5), metrics=['accuracy'])

history = stacked_ae.fit(X_train, X_train, epochs=20,

                         validation_data=(X_valid, X_valid))
from sklearn.manifold import TSNE



X_valid_compressed = stacked_encoder.predict(X_valid)

tsne = TSNE()

X_valid_2D = tsne.fit_transform(X_valid_compressed)

X_valid_2D = (X_valid_2D - X_valid_2D.min()) / (X_valid_2D.max() - X_valid_2D.min())



plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap="tab10")

plt.axis("off")

plt.show()
plt.figure(figsize=(10, 8))

cmap = plt.cm.tab10

plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap=cmap)

image_positions = np.array([[1., 1.]])

for index, position in enumerate(X_valid_2D):

    dist = np.sum((position - image_positions) ** 2, axis=1)

    if np.min(dist) > 0.02: # if far enough from other images

        image_positions = np.r_[image_positions, [position]]

        imagebox = mpl.offsetbox.AnnotationBbox(

            mpl.offsetbox.OffsetImage(X_valid[index], cmap="binary"),

            position, bboxprops={"edgecolor": cmap(y_valid[index]), "lw": 2})

        plt.gca().add_artist(imagebox)

plt.axis("off")

#save_fig("fashion_mnist_visualization_plot")

plt.show()
initial_learning_rate = 0.01

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(

    initial_learning_rate,

    decay_steps=1000,

    decay_rate=0.96,

    staircase=True)



optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
tf.random.set_seed(42)

np.random.seed(42)



conv_encoder = keras.models.Sequential([

    keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),

    keras.layers.GaussianNoise(0.2),#Adding Gaussian noise to each image 

    keras.layers.Conv2D(16, kernel_size=3, padding="SAME", activation="relu"),

    keras.layers.MaxPool2D(pool_size=2),

    keras.layers.Conv2D(32, kernel_size=3, padding="SAME", activation="relu"),

    keras.layers.MaxPool2D(pool_size=2),

    keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="relu"),

    keras.layers.MaxPool2D(pool_size=2)

])

conv_decoder = keras.models.Sequential([

    keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="valid", activation="relu",

                                 input_shape=[3, 3, 64]),

    keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="SAME", activation="selu"),

    keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="SAME", activation="sigmoid"),

    keras.layers.Reshape([28, 28])

])

conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])



conv_ae.compile(loss="binary_crossentropy", optimizer=optimizer,

                metrics=['accuracy'])

history = conv_ae.fit(X_train, X_train, epochs=15,

                      validation_data=(X_valid, X_valid))
noise_test=keras.layers.GaussianNoise(0.2)(X_test,training=True)

predicted=conv_ae.predict(noise_test[:10])
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))

for images, row in zip([noise_test[:10], predicted], axes):

    for img, ax in zip(images, row):

        ax.imshow(img, cmap='Greys_r')

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)
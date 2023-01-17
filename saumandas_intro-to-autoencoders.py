#import deep learning libraries

import tensorflow as tf

from tensorflow import keras

import tensorflow.keras.layers as L



import numpy as np

import pandas as pd
encoder = keras.Sequential()



encoder.add(L.InputLayer([784]))

encoder.add(L.Dense(500, activation='selu'))

encoder.add(L.Dense(300, activation='selu'))

encoder.add(L.Dense(2, activation='selu', name='latent'))



encoder.summary()
decoder = keras.Sequential()



decoder.add(L.InputLayer([2]))

decoder.add(L.Dense(300, activation='selu'))

decoder.add(L.Dense(500, activation='selu'))

decoder.add(L.Dense(784, activation='sigmoid', name='reconstruction'))



decoder.summary()
autoencoder = keras.Sequential([encoder, decoder])

autoencoder.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=1.5))
from tensorflow.keras.datasets.mnist import load_data



(X_train, y_train), (X_val, y_val) = load_data()

X_train = (np.array(X_train)/255.).reshape(-1, 784)

X_val = (np.array(X_val)/255.).reshape(-1, 784)
history = autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=10)
import matplotlib.pyplot as plt



img_num = 546 #experiment with values from [0, 9999]



img = X_val[img_num].reshape(28, 28)

img_recon = autoencoder.predict(X_val[img_num].reshape(1, 784))



plt.imshow(img, cmap='binary')

plt.show()

plt.imshow(img_recon.reshape(28, 28), cmap='binary')
X_val_latent = encoder.predict(X_val)
plt.figure(figsize=(30, 20))

plt.scatter(X_val_latent[:, 0], X_val_latent[:, 1], c=y_val, s=15, cmap='tab10')

plt.axis('off')
import matplotlib as mpl



plt.figure(figsize=(30, 20))

cmap = plt.cm.tab10

plt.scatter(X_val_latent[:, 0], X_val_latent[:, 1], c=y_val, s=15, cmap=cmap)

image_positions = np.array([[1., 1.]])

for index, position in enumerate(X_val_latent):

    dist = np.sum((position - image_positions) ** 2, axis=1)

    if np.min(dist) > 0.02: # if far enough from other images

        image_positions = np.r_[image_positions, [position]]

        imagebox = mpl.offsetbox.AnnotationBbox(

            mpl.offsetbox.OffsetImage(X_val[index].reshape(28, 28), cmap="binary"),

            position, bboxprops={"edgecolor": cmap(y_val[index]), "lw": 2})

        plt.gca().add_artist(imagebox)

plt.axis("off")

plt.show()
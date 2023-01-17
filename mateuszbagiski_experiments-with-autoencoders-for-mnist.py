import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, image as mpimg
from tqdm import tqdm
import random

import tensorflow as tf
from tensorflow.keras import datasets, models, layers, optimizers, regularizers, callbacks, utils


(train_X, train_y), (test_X, test_y) = datasets.mnist.load_data()
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# noise generator function

def noise_img(img, noise_ratio=.2):
    new_img = img.flatten()
    for px_i in range(len(new_img)):
        if random.random() < noise_ratio:
            new_img[px_i] = random.uniform(0, 255)
    new_img = new_img.reshape(img.shape)
    return new_img

plt.imshow(train_X[0], cmap='gray')
plt.show()

plt.imshow(noise_img(train_X[0]), cmap='gray')
plt.show()
# function to create a "noisy" dataset
def noise_data(data, noise_ratio=.2, random_state=42):
    new_data = []
    for img in tqdm(data):
        new_data.append(noise_img(img=img, noise_ratio=noise_ratio))
    return np.asarray(new_data)

train_X = train_X.reshape(*train_X.shape, 1)
test_X = test_X.reshape(*test_X.shape, 1)

train_X_noise = noise_data(train_X)
test_X_noise = noise_data(test_X)

train_X_flat = train_X.reshape(train_X.shape[0], -1)
test_X_flat = test_X.reshape(test_X.shape[0], -1)
train_X_noise.shape, test_X_noise.shape, train_X_flat.shape, test_X_flat.shape
convnet_classifier = models.Sequential(name='convnet_classifier', layers=[
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l2'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l2'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l2'),
    layers.Dense(10, activation='softmax', kernel_initializer='he_uniform', kernel_regularizer='l2')
])
convnet_classifier.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
convnet_classifier.summary()
convnet_classifier_history = convnet_classifier.fit(
    train_X, train_y,
    validation_split=.1,
    batch_size=32,
    epochs = 8,
    shuffle=True

)
convnet_classifier.evaluate(test_X, test_y)
autoencoder = models.Sequential(name='autoencoder')
autoencoder.add(layers.Input(shape=(28,28,1)))
for layer in convnet_classifier.layers[:-1]:
    autoencoder.add(layer)
for layer in autoencoder.layers:
    layer.trainable = False
autoencoder.add(layers.Dense(28*28, activation='relu'))
autoencoder.summary()
autoencoder.compile(
    optimizer=optimizers.Adam(1e-4),
    loss='mse',
    metrics=['mae']
)

autoencoder_history = autoencoder.fit(train_X, train_X_flat, epochs=6, validation_split=.1)
img_ids = np.random.randint(0, 60000, (10))
fig, axs = plt.subplots(10,4, figsize=(30,50))
for i, img_id in enumerate(img_ids):
    #print(i, img_id)
    axs[i][0].imshow(train_X[img_id].reshape(28,28), cmap='gray')
    axs[i][0].set_title('Raw original image')
    axs[i][0].tick_params(which='both', bottom=False,top=False, left=False, right=False, labelbottom=False, labelleft=False)
    axs[i][1].imshow(autoencoder.predict(train_X[img_id].reshape(1,28,28,1)).reshape(28,28), cmap='gray')
    axs[i][1].tick_params(which='both', bottom=False,top=False, left=False, right=False, labelbottom=False, labelleft=False)
    axs[i][1].set_title('Autoencoded original image')
    axs[i][2].imshow(train_X_noise[img_id].reshape(28,28), cmap='gray')
    axs[i][2].tick_params(which='both', bottom=False,top=False, left=False, right=False, labelbottom=False, labelleft=False)
    axs[i][2].set_title('Raw noisy image')
    axs[i][3].imshow(autoencoder.predict(train_X_noise[img_id].reshape(1,28,28,1)).reshape(28,28), cmap='gray')
    axs[i][3].tick_params(which='both', bottom=False,top=False, left=False, right=False, labelbottom=False, labelleft=False)
    axs[i][3].set_title('Autoencoded noisy image')

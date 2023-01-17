from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os
[file for file in os.listdir('../input') if 'kmnist' in file]
def load_data(dir):
    train_img = np.load('{}/kmnist-train-imgs.npz'.format(dir))
    train_lbl = np.load('{}/kmnist-train-labels.npz'.format(dir))
                        
    test_img = np.load('{}/kmnist-test-imgs.npz'.format(dir))
    test_lbl = np.load('{}/kmnist-test-labels.npz'.format(dir))
    
    return train_img['arr_0'], train_lbl['arr_0'], test_img['arr_0'], test_lbl['arr_0']

X_train, Y_train, X_test, Y_test = load_data('../input')
def preview_images():
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.figure(figsize=(3 * 10, 3 * 10))
    
    for i in np.arange(10):
        images = X_train[np.argwhere(Y_train == i)[:10]]
        
        for j, image in enumerate(images):
            plt.subplot(10, 10, i * 10 + j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image.reshape(28, 28))
            plt.text(28 - 3, 28 - 3, str(i), color='white', fontsize=16)

preview_images()
sns.set()
X_train = (X_train / 255.).astype('float32').reshape(X_train.shape + (1, ))
Y_train = Y_train.astype('float32')

X_test = (X_test / 255.).astype('float32').reshape(X_test.shape + (1, ))
Y_test = Y_test.astype('float32')
from tensorflow.keras import models
from tensorflow.keras import layers
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'),
    layers.Flatten(),
    layers.Dropout(.5),
    layers.Dense(100, activation='relu'),
    layers.Dropout(.2),
    layers.Dense(10, activation='softmax')
])

#optimizer = keras.optimizers.Adam(lr=0.005, decay=.0001)
optimizer = keras.optimizers.Adam()
model.compile(optimizer, 'sparse_categorical_crossentropy', metrics=['accuracy'])
checkpointer = keras.callbacks.ModelCheckpoint(filepath='mnist.h5', verbose=1, save_best_only=True)
history = model.fit(X_train, Y_train, validation_split=.2, epochs=20, callbacks=[checkpointer])
plt.figure(figsize=(16, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
from tensorflow.keras.models import load_model

model = load_model('mnist.h5')
model.evaluate(X_test, Y_test)
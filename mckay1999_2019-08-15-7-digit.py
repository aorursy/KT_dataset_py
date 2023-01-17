# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.datasets import mnist

(train_images1, train_labels1), (test_images1, test_labels1) = mnist.load_data()
#from sklearn.model_selection import train_test_split

#import numpy as np

#train = pd.read_csv("../input/train.csv")

#test = pd.read_csv("../input/test.csv")

#test_images1=test.values



#train_images1, valid_images1, train_labels1, valid_labels1 = train_test_split(train.drop(['label'], axis=1).values, train['label'].values, train_size=35000,random_state = 0)
valid_images1=train_images1[-10000:]

valid_labels1=train_labels1[-10000:]

train_images1=train_images1[:50000]

train_labels1=train_labels1[:50000]
print(train_images1.shape)

print(len(train_labels1))

print(train_labels1)

print(valid_images1.shape)

print(len(valid_labels1))

print(valid_labels1)

print(test_images1.shape)

print(len(test_labels1))

print(test_labels1)
train_images = train_images1.reshape((50000, 28, 28, 1))

train_images = train_images.astype('float32') / 255

valid_images = valid_images1.reshape((10000, 28 , 28,1))

valid_images = valid_images.astype('float32') / 255

test_images = test_images1.reshape((10000, 28 , 28,1)) 

test_images = test_images.astype('float32') / 255
from keras.utils import to_categorical

train_labels = to_categorical(train_labels1) 

valid_labels = to_categorical(valid_labels1) 

test_labels = to_categorical(test_labels1)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

          rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

          zoom_range = 0.1, # Randomly zoom image 

          width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

          height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

          )

train_gen = datagen.flow(train_images, train_labels, batch_size=128)

valid_gen = datagen.flow(valid_images, valid_labels, batch_size=128)
from keras import models 

from keras import layers

network = models.Sequential()

#network.add(layers.Input(shape=(28,28,1)))

network.add(layers.Conv2D(filters=20, kernel_size = (5, 5), activation="relu", input_shape=(28,28,1)))

network.add(layers.MaxPooling2D(pool_size=(2,2)))

network.add(layers.BatchNormalization())

network.add(layers.Dropout(0.3))

network.add(layers.Conv2D(filters=20, kernel_size = (5, 5), activation="relu"))

network.add(layers.BatchNormalization())

network.add(layers.MaxPooling2D(pool_size=(2,2)))

network.add(layers.Flatten())

network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,))) 

network.add(layers.Dropout(0.3))

network.add(layers.Dense(128, activation='relu')) 

network.add(layers.Dropout(0.3))

network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',

metrics=['accuracy'])
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

callbacks = [EarlyStopping(monitor='val_acc', patience=8, verbose=2, mode='max'),

             ReduceLROnPlateau(monitor='val_acc', factor=0.3, patience=3, verbose=2, mode='max'),

             ModelCheckpoint(monitor='val_acc', filepath='starter.hdf5', verbose=2, save_best_only=True, save_weights_only=True, mode='max')]

#history=network.fit(train_images, train_labels, validation_data=(valid_images, valid_labels), callbacks=callbacks, epochs=50, batch_size=128)

history = network.fit_generator(train_gen, epochs = 50,

                                           steps_per_epoch = train_images.shape[0] // 128,

                                           validation_steps = train_images.shape[0] // 128,

                                           validation_data = valid_gen, verbose = 1, callbacks = callbacks)
import matplotlib.pyplot as plt



# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)

plt.show()
from sklearn.metrics import confusion_matrix

pred=network.predict(test_images)

num_classes=10

cm = confusion_matrix(np.argmax(test_labels, axis=1), np.argmax(pred, axis=1))

num_classes = cm.shape[0]

count = np.unique(np.argmax(test_labels, axis=1), return_counts=True)[1].reshape(num_classes, 1)



fig = plt.figure(figsize=(10,6))

ax = plt.subplot(111)

im = ax.imshow(cm/count, cmap='YlGnBu')

im.set_clim(0, 1)

cbar = ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(num_classes))

ax.set_yticks(np.arange(num_classes))

plt.yticks(fontsize=13)

plt.xticks(fontsize=13)

for i in range(num_classes):

    for j in range(num_classes):

        text = ax.text(i, j, cm[j][i], ha="center", va="center", color="w" if (cm/count)[j, i] > 0.5 else "black", fontsize=13)

ax.set_ylabel('True Label', fontsize=16)

ax.set_xlabel('Predicted Label', fontsize=16)

ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')

plt.show()
test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)
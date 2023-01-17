import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



import keras as k

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load CIFAR10 data

(X_train, y_train), (X_test, y_test) = k.datasets.cifar10.load_data()
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
X_train = X_train / 255.0

X_test = X_test / 255.0
y_testo = y_test
y_train = to_categorical(y_train, num_classes = 10)

y_test = to_categorical(y_test, num_classes = 10)
print(y_train.shape)

print(y_test.shape)

print(y_testo.shape)
plt.imshow(X_train[5000][:,:,:])
from __future__ import print_function
model = Sequential()

batch_size = 64

num_classes = 10

epochs = 50
model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape = X_train.shape[1:]))

model.add(Conv2D(64, 5, activation='relu'))

model.add(MaxPool2D())

model.add(Dropout(0.25))

model.add(Conv2D(128, 5, activation='relu'))

model.add(MaxPool2D())

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
model.summary()
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(width_shift_range=0.1, 

                            height_shift_range = 0.1)



datagen.fit(X_train)
model.fit(X_train, y_train, batch_size=batch_size, epochs=20, validation_data = (X_test, y_test))

# model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

#                    steps_per_epoch = X_train.shape[0] // batch_size,

#                    epochs = 15, validation_data = (X_test, y_test))
predicto = model.predict_classes(X_test)

predicto.shape
y_testo.shape
from sklearn.metrics import accuracy_score

accuracy_score(y_testo, predicto)
name = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
from mpl_toolkits.axes_grid1 import ImageGrid
R = 3

C = 4

fig, axes = plt.subplots(R,C, figsize=(16,10))



for i in range(R):

    for j in range(C):

        r = np.random.randint(10000, size=1)[0]

        axes[i, j].imshow(X_test[r][:,:,:])

        axes[i, j].plot()

        #print('this is a', name[y_testo[r][0]], '-------- prediction is:', name[predicto[r]])

        axes[i, j].text(0, 0, 'Prediction: %s' % name[predicto[r]], color='w', backgroundcolor='k', alpha=0.8)

        axes[i, j].text(0, 3.9, 'LABEL: %s' % name[y_testo[r][0]], color='k', backgroundcolor='w', alpha=0.8)

        axes[i, j].axis('off')

        
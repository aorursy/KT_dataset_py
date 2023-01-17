import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))
import cv2

import matplotlib.pyplot as plt 

import seaborn as sns

import os

from PIL import Image

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

from keras.utils import np_utils
import tensorflow as tf



tf.__version__
parasitized_data = os.listdir('../input/cell_images/cell_images/Parasitized/')

print(parasitized_data[:10]) 



uninfected_data = os.listdir('../input/cell_images/cell_images/Uninfected/')

print('\n')

print(uninfected_data[:10])
plt.figure(figsize = (12,12))

for i in range(4):

    plt.subplot(1, 4, i+1)

    img = cv2.imread('../input/cell_images/cell_images/Parasitized' + "/" + parasitized_data[i])

    plt.imshow(img)

    plt.title('PARASITIZED : 1')

    plt.tight_layout()

plt.show()
plt.figure(figsize = (12,12))

for i in range(4):

    plt.subplot(1, 4, i+1)

    img = cv2.imread('../input/cell_images/cell_images/Uninfected' + "/" + uninfected_data[i+1])

    plt.imshow(img)

    plt.title('UNINFECTED : 0')

    plt.tight_layout()

plt.show()
data = []

labels = []

for img in parasitized_data:

    try:

        img_read = plt.imread('../input/cell_images/cell_images/Parasitized/' + "/" + img)

        img_resize = cv2.resize(img_read, (50, 50))

        img_array = img_to_array(img_resize)

        data.append(img_array)

        labels.append(1)

    except:

        None

        

for img in uninfected_data:

    try:

        img_read = plt.imread('../input/cell_images/cell_images/Uninfected' + "/" + img)

        img_resize = cv2.resize(img_read, (50, 50))

        img_array = img_to_array(img_resize)

        data.append(img_array)

        labels.append(0)

    except:

        None

plt.imshow(data[0])

plt.show()
image_data = np.array(data)

labels = np.array(labels)
idx = np.arange(image_data.shape[0])

np.random.shuffle(idx)

image_data = image_data[idx]

labels = labels[idx]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(image_data, labels, test_size = 0.2, random_state = 101)
y_train = np_utils.to_categorical(y_train, num_classes = 2)

y_test = np_utils.to_categorical(y_test, num_classes = 2)
print(f'SHAPE OF TRAINING IMAGE DATA : {x_train.shape}')

print(f'SHAPE OF TESTING IMAGE DATA : {x_test.shape}')

print(f'SHAPE OF TRAINING LABELS : {y_train.shape}')

print(f'SHAPE OF TESTING LABELS : {y_test.shape}')
import keras

from keras.layers import Dense, Conv2D

from keras.layers import Flatten

from keras.layers import MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Activation

from keras.layers import BatchNormalization

from keras.layers import Dropout

from keras.models import Sequential

from keras import backend as K

from keras import optimizers
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
model = Sequential()



model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (50,50,3)))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.2))



model.add(Conv2D(32, (3,3), activation = 'relu'))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.2))



model.add(Conv2D(32, (3,3), activation = 'relu'))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.2))



model.add(Flatten())



model.add(Dense(512, activation = 'relu'))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.5))

model.add(Dense(2, activation = 'softmax'))



model.summary()
model.compile(optimizer = 'sgd',

                    loss = 'sparse_categorical_crossentropy',

                    metrics = ['accuracy'])
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
h = model.fit(x_train, y_train, epochs = 20, batch_size = 32)
plt.figure(figsize = (18,8))

plt.plot(range(20), h.history['acc'], label = 'Training Accuracy')

plt.plot(range(20), h.history['loss'], label = 'Taining Loss')

#ax1.set_xticks(np.arange(0, 31, 5))

plt.xlabel("Epoch's")

plt.ylabel('Accuracy/Loss Value')

plt.title('Training Accuracy and Training Loss')

plt.legend(loc = "best")

predictions = model.evaluate(x_test, y_test)

print(f'LOSS : {predictions[0]}')

print(f'ACCURACY : {predictions[1]}')
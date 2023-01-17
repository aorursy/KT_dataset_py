# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import os

import cv2

from keras.utils import np_utils

from tqdm import tqdm
# Loading data method

def load_data(path):

    labels = []

    images = []

    size = (150,150)

    for i in tqdm(os.listdir(path)):

        folder = path + '/' + i

        for j in os.listdir(folder):

            img_path = folder + '/' + j

            temp_img = cv2.imread(img_path)

            temp_img = cv2.resize(temp_img, size)

            temp_label = i

            images.append(temp_img)

            labels.append(temp_label)

    images = np.array(images, dtype = 'float32')/255

    labels = np.array(labels)

    return images, labels
# Calling load_data method

x_train, y_train = load_data('/kaggle/input/intel-image-classification/seg_train/seg_train')

x_test, y_test = load_data('/kaggle/input/intel-image-classification/seg_test/seg_test')
# Plotting random 25 images

plt.figure(figsize=(25,25))

for i in range(1,26):

    random = np.random.randint(x_train.shape[0])

    plt.subplot(5,5,i)

    plt.imshow(x_train[random])

    plt.title(y_train[random])
# Checking if each class contains equal number of images

labels = os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train')

label_count = {}

for i in labels:

    path = os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train/' + i)

    label_count.update({i:len(path)})

plt.bar(label_count.keys(),label_count.values())
# One hot encoding labels

def one_hot(data):

    y = []

    for nb, lb in enumerate(labels):

        for i in data:

            if i == lb:

                y.append(nb)

    y = np_utils.to_categorical(y, len(labels))

    return y

y_train = one_hot(y_train)

y_test = one_hot(y_test)
# Shuffling training and testing data

from sklearn.utils import shuffle

x_train, y_train = shuffle(x_train, y_train)

x_test, y_test = shuffle(x_test, y_test)
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=23)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=45, width_shift_range=0.2, height_shift_range=0.1, horizontal_flip=True)

datagen.fit(x_train)
# Importing Keras Libraries

from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dense, Activation, Flatten, Dropout, BatchNormalization

from keras.callbacks import ModelCheckpoint

from keras.applications.vgg16 import VGG16

from keras.regularizers import Regularizer,l2

from keras.applications.resnet50 import ResNet50

from keras.applications.mobilenet import MobileNet

import warnings

warnings.filterwarnings('ignore')
model1 = Sequential()

model1.add(Conv2D(64, (3,3), activation='relu', input_shape=x_train.shape[1:]))

model1.add(Conv2D(128,(3,3), activation='relu'))

model1.add(MaxPool2D(5,5))

model1.add(Conv2D(150,(3,3), activation='relu'))

model1.add(MaxPool2D(5,5))

model1.add(Flatten())

model1.add(Dense(128, activity_regularizer=l2(0.001)))

model1.add(Dropout(0.3))

model1.add(Dense(len(labels), activation='softmax'))

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint1 = [ModelCheckpoint('intel_weights1.hdf5', save_best_only=True, monitor='val_loss', mode='auto', verbose=1)]

model1.summary()
model1.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs = 20, steps_per_epoch=x_train.shape[0]//32, validation_data=(x_valid,y_valid), verbose = 1, callbacks=checkpoint1)
model1.load_weights('intel_weights1.hdf5')

score1 = model1.evaluate(x_test,y_test)

print('Accuracy for self created model: ', score1[1]*100)
model2 = Sequential()

vgg = VGG16(weights='imagenet', include_top=False, input_shape=x_train.shape[1:])

vgg.trainable=False

model2.add(vgg)

model2.add(Conv2D(128,(3,3), activation='relu'))

model2.add(Flatten())

model2.add(Dense(128,activation='relu',  activity_regularizer=l2(0.001)))

model2.add(Dropout(0.3))

model2.add(Dense(len(labels), activation='softmax'))

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint2 = [ModelCheckpoint('intel_weights2.hdf5', save_best_only=True, monitor='val_loss', mode='auto', verbose=1)]

model2.summary()
model2.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs = 10, steps_per_epoch=x_train.shape[0]//32, validation_data=(x_valid,y_valid), verbose = 1, callbacks=checkpoint2)
model2.load_weights('intel_weights2.hdf5')

score2 = model2.evaluate(x_test,y_test)

print('Accuracy for VGG16 model: ', score2[1]*100)
model3 = Sequential()

resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=x_train.shape[1:])

resnet50.trainable=False

model3.add(resnet50)

model3.add(Conv2D(128,(3,3), activation='relu'))

model3.add(Flatten())

model3.add(Dense(128,activation='relu',  activity_regularizer=l2(0.001)))

model3.add(Dropout(0.3))

model3.add(Dense(128,activation='relu'))

model3.add(Dropout(0.2))

model3.add(Dense(len(labels), activation='softmax'))

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint3 = [ModelCheckpoint('intel_weights3.hdf5', save_best_only=True, monitor='val_loss', mode='auto', verbose=1)]

model3.summary()
model3.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs = 10, steps_per_epoch=x_train.shape[0]//32, validation_data=(x_valid,y_valid), verbose = 1, callbacks=checkpoint3)
model3.load_weights('intel_weights3.hdf5')

score3 = model3.evaluate(x_test,y_test)

print('Accuracy for ResNet model: ', score3[1])
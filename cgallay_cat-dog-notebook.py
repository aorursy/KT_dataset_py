import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

from random import shuffle



import os

import cv2
train_path = "../input/train"



ROWS = 128

COLS = 128

CHANNELS = 3



images = [img for img in os.listdir(train_path)]

images_dog = [img for img in os.listdir(train_path) if "dog" in img]

images_cat = [img for img in os.listdir(train_path) if "cat" in img]



#only taking a subset (less accuracy but faster training)

train_dog = images_dog[:1000]

train_cat = images_cat[:1000]

valid_dog = images_dog[1000:1100]

valid_cat = images_cat[1000:1100]



train_list = train_dog + train_cat

valid_list = valid_dog + valid_cat



shuffle(train_list)



train = np.ndarray(shape=(len(train_list),ROWS, COLS))

labels = np.ndarray(len(train_list))



for i, img_path in enumerate(train_list):

    img = cv2.imread(os.path.join(train_path, img_path), 0)

    img = cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

    

    train[i] = img

    if "dog" in img_path:

        labels[i] = 0

    else:

        labels[i] = 1



valid = np.ndarray(shape=(len(valid_list), ROWS, COLS))

valid_labels = np.ndarray(len(valid_list))



for i, img_path in enumerate(valid_list):

    img = cv2.imread(os.path.join(train_path, img_path), 0)

    img = cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

    

    valid[i] = img

    if "dog" in img_path:

        valid_labels[i] = 0

    else:

        valid_labels[i] = 1
n = 33

plt.subplot(1,2,1)

plt.imshow(cv2.imread(os.path.join(train_path, train_list[len(train_list)-n])))

plt.subplot(1,2,2)

plt.imshow(train[len(train_list)-n])

plt.title(labels[len(train_list)-n])

plt.show()
def average(data):

    minimum = np.min(data)

    maximum = np.max(data)

    data = (data-minimum)/(maximum-minimum)

    data = data - np.mean(data)

    return data



train = average(train)
from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import Dense, Activation, Flatten, Dropout, MaxPooling2D

from keras.regularizers import l2



model = Sequential()

model.add(Convolution2D(8, 3, 3, border_mode='same', input_shape=(ROWS, COLS, 1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='same'))

model.add(Convolution2D(4, 3, 3, border_mode='same', activation='relu'))

model.add(Flatten())

model.add(Dense(output_dim=64, W_regularizer=l2(0.01)))

model.add(Dropout(0.5))

model.add(Activation('relu'))

model.add(Dense(output_dim=2, W_regularizer=l2(0.01)))  #binary classification

model.add(Activation('softmax'))
from keras.utils.np_utils import to_categorical

from keras.optimizers import SGD, RMSprop

#model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.005, momentum=0.9, decay=0.1, nesterov=False), metrics=['accuracy'])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])

labels_ = to_categorical(labels, 2) #convert labels to a matrix representation 

train_ = np.resize(train, (len(train), ROWS, COLS, 1))

model.fit(train_, labels_, nb_epoch=20, batch_size=32)
valid_labels_ = to_categorical(valid_labels, 2)

valid_ = average(valid)

valid_ = np.resize(valid_, (len(valid_), ROWS, COLS, 1))

print("valid set :", model.evaluate(valid_, valid_labels_, verbose=False)[1]*100, "%")

print("--------------------")

print("train set :", model.evaluate(train_, labels_, verbose=False)[1]*100, "%")
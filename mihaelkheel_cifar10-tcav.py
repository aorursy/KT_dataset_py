from tensorflow.python.client import device_lib

device_lib.list_local_devices()
!git clone https://github.com/pnxenopoulos/cav-keras.git

!cp -R cav-keras/cav cav/
import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



import keras

from keras.datasets import cifar100, cifar10

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D



import sys

import os

sys.path.insert(0, os.path.abspath('../..'))



from cav.tcav import *



np.random.seed(1996)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()



# Keep ships (8) from CIFAR-10

interested_class = y_train == [8]

interested_class_indx = [i for i, x in enumerate(interested_class) if x]

x_train_class_one = x_train[interested_class_indx]

other = y_train == [2]

other_indx = [i for i, x in enumerate(other) if x]

x_train_class_two = x_train[other_indx]



x_train = np.append(x_train_class_one, x_train_class_two, axis = 0)

y_train = [1] * 5000

y_train = y_train + [0] * 5000



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255
f, axarr = plt.subplots(1,2)

axarr[0].imshow(x_train[0])

axarr[1].imshow(x_train[7777])
(x_train_concept, y_train_concept), (x_test_concept, y_test_concept) = cifar100.load_data()



# keep sea (71) from CIFAR-100

concept = y_train_concept == [71]

indices = concept

indx_to_use = [i for i, x in enumerate(indices) if x]



x_train_concept = x_train_concept[indx_to_use]
batch_size = 32

epochs = 5



model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))



# initiate optimizer

opt = keras.optimizers.Adam(lr=0.001)



# train the model

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])



model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
model.summary()
tcav_obj = TCAV()

tcav_obj.set_model(model)
tcav_obj.split_model(bottleneck = 1, conv_layer = True)

tcav_obj.train_cav(x_train_concept)

tcav_obj.calculate_sensitivity(x_train, y_train)

tcav_obj.print_sensitivity()
tcav_obj.split_model(bottleneck = 5, conv_layer = True)

tcav_obj.train_cav(x_train_concept)

tcav_obj.calculate_sensitivity(x_train, y_train)

tcav_obj.print_sensitivity()
tcav_obj.split_model(bottleneck = 7, conv_layer = True)

tcav_obj.train_cav(x_train_concept)

tcav_obj.calculate_sensitivity(x_train, y_train)

tcav_obj.print_sensitivity()
tcav_obj.split_model(bottleneck = 11, conv_layer = True)

tcav_obj.train_cav(x_train_concept)

tcav_obj.calculate_sensitivity(x_train, y_train)

tcav_obj.print_sensitivity()
CIFAR100_LABELS_LIST = [

    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 

    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 

    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 

    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 

    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 

    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',

    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',

    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',

    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',

    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',

    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',

    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',

    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',

    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',

    'worm'

]

labels = {x:i for i,x in enumerate(CIFAR100_LABELS_LIST)}
CIFAR100_LABELS_LIST[71]

labels['road'], labels['forest'], labels["sea"]

(x_train, y_train), (x_test, y_test) = cifar10.load_data()



# Keep airplanes from CIFAR-10

interested_class = y_train == [2]

interested_class_indx = [i for i, x in enumerate(interested_class) if x]

x_train_class_one = x_train[interested_class_indx]

other = y_train == [6]

other_indx = [i for i, x in enumerate(other) if x]

x_train_class_two = x_train[other_indx]



x_train = np.append(x_train_class_one, x_train_class_two, axis = 0)

y_train = [0] * 5000

y_train = y_train + [1] * 5000



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255



f, axarr = plt.subplots(1,2)

axarr[0].imshow(x_train[11])

axarr[1].imshow(x_train[7777])
(x_train_concept, y_train_concept), (x_test_concept, y_test_concept) = cifar100.load_data()



# keep cloud (23) from CIFAR-100

concept = y_train_concept == [23]

indices = concept

indx_to_use = [i for i, x in enumerate(indices) if x]



x_train_concept = x_train_concept[indx_to_use]
# Set parameters

batch_size = 32

epochs = 20



model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))



# initiate optimizer

opt = keras.optimizers.Adam(lr=0.001)



# train the model

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])



model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
tcav_obj = TCAV()

tcav_obj.set_model(model)
tcav_obj.split_model(bottleneck = 1, conv_layer = True)

tcav_obj.train_cav(x_train_concept)

tcav_obj.calculate_sensitivity(x_train, y_train)

tcav_obj.print_sensitivity()
tcav_obj.split_model(bottleneck = 5, conv_layer = True)

tcav_obj.train_cav(x_train_concept)

tcav_obj.calculate_sensitivity(x_train, y_train)

tcav_obj.print_sensitivity()
tcav_obj.split_model(bottleneck = 7, conv_layer = True)

tcav_obj.train_cav(x_train_concept)

tcav_obj.calculate_sensitivity(x_train, y_train)

tcav_obj.print_sensitivity()
tcav_obj.split_model(bottleneck = 11, conv_layer = True)

tcav_obj.train_cav(x_train_concept)

tcav_obj.calculate_sensitivity(x_train, y_train)

tcav_obj.print_sensitivity()
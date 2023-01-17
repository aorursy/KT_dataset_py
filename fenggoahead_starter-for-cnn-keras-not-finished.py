# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# -*- coding: utf-8 -*-

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import numpy as np

import pandas as pd

import cv2



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



master = pd.read_csv("../input/train_labels.csv")

master.head()

img_path = "../input/train/"

y = []

file_paths = []

for i in range(100): 

    #At firet using only small dataset, finally, change 100 to "len(master)"

    file_paths.append( img_path + str(master.ix[i][0]) +'.jpg' )

    y.append(master.ix[i][1]) #save the path for every img 

y = np.array(y)

image_num_x = 128 #At first resize image to a smaller one.

image_num_y = image_num_x*886//1154

tile_size=(image_num_x,image_num_y)
x = []

for i, file_path in enumerate(file_paths):

    img = cv2.imread(file_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resize = cv2.resize(img, dsize=tile_size)

    x.append(img_resize)
x = np.array(x)

data_num = len(y)

random_index = np.random.permutation(data_num)

x_shuffle = []

y_shuffle = []

for i in range(data_num):

    x_shuffle.append(x[random_index[i]])

    y_shuffle.append(y[random_index[i]])

x = np.array(x_shuffle) 

y = np.array(y_shuffle)



x = (x - np.mean(x))/np.std(x)



val_split_num = int(round(0.2*len(y))) # 20% for cross validation

x1_train = x[val_split_num:]

y1_train = y[val_split_num:]

x1_cross = x[:val_split_num]

y1_cross = y[:val_split_num]



print('x1_train', x1_train.shape)

print('y1_train', y1_train.shape)

print('x1_cross', x1_cross.shape)

print('y1_cross', y1_cross.shape)



x1_train = x1_train.astype('float32')

x1_cross = x1_cross.astype('float32')
# Starting architecture

from keras.models import Sequential



from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.layers import Flatten, Dense, Dropout

from keras.layers import Conv2D, MaxPool2D, BatchNormalization



from keras.optimizers import Adam



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding







from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler
model = Sequential()



model.add(ZeroPadding2D((1, 1), input_shape=(image_num_y, image_num_x,3)))

model.add(Convolution2D(32, 3, 3, activation='relu'))

model.add(BatchNormalization())

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(32, 3, 3, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Dropout(0.25))



model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(32, 3, 3, activation='relu'))

model.add(BatchNormalization())

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(32, 3, 3, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Dropout(0.25))



model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(BatchNormalization())

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



print(model.summary())
model.compile(loss='binary_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = 0.1,

                            width_shift_range = 0.1,

                            rotation_range = 10)

hist = model.fit_generator(datagen.flow(x1_train, y1_train, batch_size=16),

                           steps_per_epoch=20,

                           epochs=200, #Increase this when not on Kaggle kernel

                           verbose=1,  #1 for ETA, 0 for silent

                           validation_data=(x1_cross, y1_cross), #For speed

                           callbacks=[annealer])

acc = model.evaluate(x1_cross, y1_cross)

print('Evaluation accuracy:{0}'.format(round(acc, 4)))
import os

import cv2

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Conv2D, Flatten

from keras.layers import MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Activation, BatchNormalization

from keras.layers import Dropout
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
parasitized_data = os.listdir('../input/cell_images/Parasitized_VGG/')

uninfected_data = os.listdir('../input/cell_images/Uninfected_VGG/')
# images_1 = os.listdir(base_address_1)

# images_0 = os.listdir(base_address_0)

# # Removing 'Thumbs.db' file from the list of images

# for img in images_1:

#     if 'Thumbs.db' in img:

#         idx = images_1.index(img)

#         images_1.pop(idx)

# for img in images_0:

#     if 'Thumbs.db' in img:

#         idx = images_0.index(img)

#         images_0.pop(idx)
print('The image files for Uninfected are: '+str(len(uninfected_data)))

print('The image files for Infected are: '+str(len(parasitized_data)))
data = []

labels = []

for img in parasitized_data:

    try:

        img_array = cv2.imread('../input/cell_images/Parasitized_VGG/' + img)

        data.append(img_array)

        labels.append(1)

    except:

        print("Runtime Exception due to image at index "+ str(parasitized_data.index(img)))

        

for img in uninfected_data:

    try:

        img_array = cv2.imread('../input/cell_images/Uninfected_VGG' + "/" + img)

        data.append(img_array)

        labels.append(0)

    except:

        print("Runtime Exception due to image at index "+ str(uninfected_data.index(img)))
print(len(data))

print(len(labels))
# Visualization an uninfected image

img = cv2.imread('../input/cell_images/Uninfected_VGG' + "/" + uninfected_data[0])

plt.imshow(img)
# Visualization an infected image

img = cv2.imread('../input/cell_images/Parasitized_VGG' + "/" + parasitized_data[0])

plt.imshow(img)
image_data = np.array(data)

labels = np.array(labels)

# Shuffling the data 

idx = np.arange(image_data.shape[0])

np.random.shuffle(idx)

image_data = image_data[idx]

labels = labels[idx]
# Training and Test data split

X_train, X_test, y_train, y_test = train_test_split(image_data,labels, test_size=0.20, random_state = 101) 
# One- Hot Encoding the lables

y_train = np_utils.to_categorical(y_train, num_classes = 2)

y_test = np_utils.to_categorical(y_test, num_classes = 2)
# Normalizing the data

X_train = X_train.astype('float32')/255

X_test = X_test.astype('float32')/255
########################################################################################################################

# Defining the model

###########################################################################################################################



model = Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.3))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2, activation = 'softmax'))
model.summary()
# Defining the optimizer, loss, performance of the metrcis of the model

batches = 50

optim = optimizers.Adam(lr = 0.001, decay = 0.001 / batches)

model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=batches,epochs=25,verbose=1)
model.evaluate(X_test,y_test, steps = 1)
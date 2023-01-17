# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from keras.preprocessing.image import ImageDataGenerator

from keras import layers

from keras import Sequential

from keras import models

from keras import optimizers

from sklearn.model_selection import train_test_split # split the data into train and test set

import seaborn as sns

import cv2

import numpy as np

import os

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import random

import gc

import matplotlib.image as mpimg

import csv

from tqdm import tqdm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_dir = "/kaggle/input/ieeeensiai/train/train"

test_dir = "/kaggle/input/ieeeensiai/test/test"



train_set = ["/kaggle/input/ieeeensiai/train/train/{}".format(i) for i in os.listdir(train_dir)] # get the image training set

test_set = ["/kaggle/input/ieeeensiai/test/test/{}".format(i) for i in os.listdir(test_dir)] # get the image test set

train_set= train_set[:11000]

test_set= test_set[:5000]

dict_labels = {'10':0,'20':1,'50':2,'100':3,'200':4,'500':5,'1000':6,'2000':7,'5000':8}

random.shuffle(train_set)
# displaying image data

def display_data(image):

    img=mpimg.imread(image)

    imgplot = plt.imshow(img)

    plt.show()

display_data(train_set[8])
labels = {}

with open('/kaggle/input/ieeeensiai/train.csv') as csv_file:

    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:

        if row[0]!= 'img':

            labels[row[0]] = row[1]
def process_images(list_of_images, labels):

    X = []

    Y = []

#     vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

    for path in tqdm(list_of_images):

        im = cv2.imread(path)

        im = im[:,::-1]

        X.append(im)

#         im = im - vgg_mean

        if labels:

            label = path.split('/')

            label = label[-1]

            label = label.split('.')[0]

            categ = [0.]*9

            categ[dict_labels[labels[label]]] = 1.

            Y.append(categ)

    return X, Y
# processing data and making labels

X, Y = process_images(train_set, labels)
# delete unnecessary data

del train_set

gc.collect()
# convert to array

X = np.array(X)

Y = np.array(Y)
# test image with corresponding label

def print_label(y):

    for key, value in dict_labels.items():

        index = list(y).index(1.)

        if index == value:

            return(key)

# testing the labeled data

plt.imshow(X[6])

plt.title(print_label(Y[6]))
print('image data shape', X.shape)

print('label shape', Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.20)



print("shape of train images is :",X_train.shape)

print("shape of validation images is :",X_val.shape)

print("shape of train labels is :",y_train.shape)

print("shape of validation labels is :",y_train.shape)
# clear memory

del X

del Y

gc.collect()



# get the lenght of train and validation data

ntrain = len(X_train)

nval = len(X_val)



# bach_size

batch_size = 64
model = models.Sequential()

# Conv Block 1



model.add(layers.Conv2D(64, (3, 3), input_shape=(224,224,3), activation='relu', padding='same'))



model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))



model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))







# Conv Block 2



model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))



model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))



model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))







# Conv Block 3



model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))



model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))



model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))



model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))







# Conv Block 4



model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))



model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))



model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))



model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))







# Conv Block 5



model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))



model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))



model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))



model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))







# FC layers



model.add(layers.Flatten())



model.add(layers.Dense(4096, activation='relu'))



model.add(layers.Dense(4096, activation='relu'))



model.add(layers.Dense(9, activation='softmax'))
model.summary()
# compile the model

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
# create the augmentation configuration

# this helps prevent overfitting, since we use a small dataset

train_datagen = ImageDataGenerator(rescale=1./255,

                                    rotation_range=40,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    shear_range = 0.2,

                                    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255) # for validation dataset
# creat the image generators

train_generator = train_datagen.flow(X_train, y_train , batch_size=batch_size)

val_generator = val_datagen.flow(X_val, y_val , batch_size=batch_size)
# the training part

# we train for 64 epochs

history = model.fit_generator(train_generator,

                             steps_per_epoch=ntrain // batch_size,

                             epochs=32,

                             validation_data=val_generator,

                             validation_steps=nval // batch_size)
#Save the model

# model.save('model_keras.h5')
# process the test set 

X_test,_ = process_images(test_set[:1000], labels=[])

x = np.array(X_test)

test_datagen = ImageDataGenerator(rescale=1./255)

x_test = test_datagen.flow(x, batch_size=1)

predected = model.predict_generator(x_test,steps=len(x_test))

print(len(predected))
# make predection

predect = []

for pre in predected:

    index = list(pre).index(max(list(pre)))

    for key,value in dict_labels.items():

        if value ==index:

            predect.append(key)
# print some predection

ind = 40

pred = model.predict(x_test[ind])

list_label = list(pred[0,])

print(list_label)

maximum = max(list_label)

index = list_label.index(maximum)

text_label = ''

for key,val in dict_labels.items():

    if val ==index:

        text_label=key

plt.title('it is a '+ text_label +" coin")

imgplot = plt.imshow(x_test[ind][0])

plt.show()
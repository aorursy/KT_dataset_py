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
print(os.listdir("../input/flowers/flowers"))


from skimage import io

from scipy.misc import imread, imresize, imshow

import random

import glob

import matplotlib.pyplot as plt

% matplotlib inline



path = "../input/flowers/flowers"

flowers_classes = ['daisy', 'sunflower', 'tulip', 'rose', 'dandelion']

# Lets visualize one random picture from each type of flower



for i, type in enumerate(flowers_classes):

    flower_path = os.path.join(path, flowers_classes[i], '*')

    flower_path = glob.glob(flower_path)

    rand_index = random.randint(0, len(flower_path))

    image = io.imread(flower_path[rand_index])

    size = image.shape

    plt.xlabel(type+" size: "+str(size[0])+" "+str(size[1]))

    plt.imshow(image)

    plt.show()
IMG_SIZE = 150   ## Image size

X = [] 

Y = []
## Number of image for each type



for idx, type in enumerate(flowers_classes):

    flower_path = os.path.join(path, flowers_classes[idx], '*')

    flower_path = glob.glob(flower_path)

    for i in range(len(flower_path)):

        try:

            image = io.imread(flower_path[i])

            image = imresize(image, (IMG_SIZE, IMG_SIZE))

            X.append(image)

            Y.append(type)

        except:

            print(flower_path[i])

            continue

        

rand_index = random.randint(0, len(X))

image = X[rand_index]

label = Y[rand_index] 

plt.imshow(image)

plt.suptitle(label)
# function to plot n images using subplots

def plot_image(images, captions=None, cmap=None ):

    f, axes = plt.subplots(1, len(images), sharey=True)

    f.set_figwidth(15)

    for ax,image in zip(axes, images):

        ax.imshow(image, cmap)
# plotting the original image and the RGB channels

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)

f.set_figwidth(15)

ax1.imshow(image)



# RGB channels

ax2.imshow(image[:, : , 0])

ax3.imshow(image[:, : , 1])

ax4.imshow(image[:, : , 2])

f.suptitle('Different Channels of Image')
image_after_normalization = image/255



plot_image([image, image_after_normalization], cmap='gray')
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical



le=LabelEncoder()

y=le.fit_transform(Y)

y=to_categorical(y,5)

## Create train. validation and test samples from input.

from sklearn.model_selection import train_test_split

X = np.array(X)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)



## Lets takeout some data from training data as validation data

x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1,random_state=42)
print(x_train[0].shape)

print(y_train[0].shape)

print(x_val[0].shape)

print(y_val[0].shape)

print(x_test[0].shape)

print(y_test[0].shape)
from keras.models import Sequential, Model

from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras import optimizers

from keras.layers import Dense, Dropout, Flatten

from keras.optimizers import Adam
# Define model

model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))

model.add(Conv2D(32, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3,3), activation='relu'))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3,3), activation='relu'))

model.add(Conv2D(128, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.25))



# softmax layer

model.add(Dense(5, activation='softmax'))



# model summary

model.summary()
from keras.optimizers import Adam

optimiser = Adam()

model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
batch_size=32

epochs=50
from keras.preprocessing.image import ImageDataGenerator



# Adding rescale, rotation_range, width_shift_range, height_shift_range,

# shear_range, zoom_range, and horizontal flip to our ImageDataGenerator

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.4,

    height_shift_range=0.4,

    shear_range=0.2,

    zoom_range=0.3,

    horizontal_flip=True

)



# Note that the validation data should not be augmented!

val_datagen = ImageDataGenerator(

    rescale=1./255

)



# Flow training images in batches of 32 using train_datagen generator

train_generator = train_datagen.flow(

    x_train,

    y_train,

    batch_size=batch_size

)



val_generator = val_datagen.flow(

    x_val,

    y_val,

    batch_size=batch_size

)
model_name = 'model' + '/'

    

if not os.path.exists(model_name):

    os.mkdir(model_name)

        

filepath = model_name + 'model.h5'



checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)



LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1, verbose=1)

callbacks_list = [checkpoint, LR]
model_hist = model.fit_generator(train_generator, steps_per_epoch=len(x_train)/batch_size, epochs=epochs, verbose=1, 

                    callbacks=callbacks_list, validation_data=val_generator, 

                    validation_steps=len(x_val)/batch_size, class_weight=None, workers=1, initial_epoch=0)
plt.plot(model_hist.history['loss'])

plt.plot(model_hist.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
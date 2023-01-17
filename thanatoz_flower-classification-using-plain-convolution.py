import os

import sys

from collections import Counter

import json

import cv2

import numpy as np

import pandas as pd

from tqdm.notebook import tqdm

from matplotlib import pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split



from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.utils import to_categorical
# Location of our dataset

!ls ../input

!ls ../input/flower-classification-dataset/

# !ls ../input/flower-classification-dataset/files/
# What is inside the labels.csv

ROOT_PATH = "../input/flower-classification-dataset/"

df = pd.read_csv(os.path.join(ROOT_PATH, 'labels.csv'))

df.tail()
X = []

y = []

for image, label in tqdm(zip(df.image_id.values, df.category.values), total=len(df)):

    try:

        xt = np.array(Image.open(os.path.join(ROOT_PATH, f"files/{image}.jpg")).resize((128,128)))

        yt = label

        X.append(xt)

        y.append(yt)

    except:

        print(os.path.join(ROOT_PATH, f"files/{image}.jpg"))

    

X = np.array(X)

y = np.array(y)

X.shape, y.shape
files = os.listdir(os.path.join(ROOT_PATH, 'files'))

print(f"The total number of files in the dataset are {len(files)}")
# Lets see that the number of images in the dataset equals to the provided labels

print(f"The total number of points in the labels.csv are {len(df)}")
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_y = to_categorical(train_y)

test_y = to_categorical(test_y)

train_y.shape, test_y.shape
train_X.shape, train_y.shape
# plot first few images

plt.figure(figsize=(12,12))

for i in range(9):

    # define subplot

    plt.subplot(330 + 1 + i)

    # plot raw pixel data

    plt.imshow(train_X[i])

# show the figure

plt.show()
plt.figure(figsize=(18,6))

df["category"].value_counts().plot(kind='bar')
heights = []

widths = []

for image in tqdm(os.listdir(os.path.join(ROOT_PATH, "files"))):

    ht, wt = Image.open(os.path.join(ROOT_PATH, f"files/{image}")).size

    heights.append(ht)

    widths.append(wt)
Counter(heights), Counter(widths)
!rm -rf preview

!mkdir preview
datagen = ImageDataGenerator(

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')



img = load_img(os.path.join(ROOT_PATH, f"files/0.jpg"))  # this is a PIL image

x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)

x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)



# the .flow() command below generates batches of randomly transformed images

# and saves the results to the `preview/` directory

i = 0

for batch in datagen.flow(x, batch_size=1,

                          save_to_dir="preview", save_prefix='f', save_format='jpg'):

    i += 1

    if i > 20:

        break  # otherwise the generator would loop indefinitely
!ls preview
x=[]

for image in os.listdir('preview'):

    xt = np.array(Image.open(os.path.join("preview", image)).resize((128,128)))

    x.append(xt)    

    

# plot first few images

plt.figure(figsize=(12,12))

for i in range(9):

    # define subplot

    plt.subplot(330 + 1 + i)

    # plot raw pixel data

    plt.imshow(x[i])

# show the figure

plt.show()
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(103))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
batch_size = 16



# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(rescale=1./255)
# this is a generator that will read pictures found in

# subfolers of 'data/train', and indefinitely generate

# batches of augmented image data

train_generator = train_datagen.flow(

        train_X,

        train_y,

        batch_size=batch_size,

        shuffle=True

        )  # since we use binary_crossentropy loss, we need binary labels



# this is a similar generator, for validation data

validation_generator = test_datagen.flow(

        test_X,

        test_y,

        shuffle=False,

        )
hist = model.fit_generator(

        train_generator,

        epochs=50,

        validation_data=validation_generator)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/test_set/test_set"))


# Any results you write to the current directory are saved as output.
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,array_to_img,load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
from os.path import join
image_size=4000
dog_img_dir = "../input/training_set/training_set/dogs/"

cat_img_dir = "../input/training_set/training_set/cats/"
list_dog = os.listdir("../input/training_set/training_set/dogs/")
dog_img_path0=[join(dog_img_dir,dog ) for dog in list_dog]
dog_img_path=dog_img_path0[0:5]
list_cat = os.listdir("../input/training_set/training_set/cats/")
cat_img_path0=[join(cat_img_dir,cat ) for cat in list_cat]
cat_img_path=cat_img_path0[0:5]
len(dog_img_path)

def read_prep_images(image_paths,img_height=image_size,img_width=image_size):
    img1 = [load_img(dogs,target_size=(img_height,img_width)) for dogs in dog_img_path]
    image_arry = np.array([img_to_array(image) for image in img1])
    output = preprocess_input(image_arry)
    return(output)
dog_size = read_prep_images(dog_img_path)

cat_size = read_prep_images(cat_img_path)
cat_dog = np.concatenate((dog_size,cat_size),axis=0)
cat_dog.shape
dog_size.shape
num_classes=2
model = Sequential()
model.add(Conv2D(10,kernel_size=(3,3),strides=2,activation='relu',
                 input_shape=(150,150,3)))
model.add(Conv2D(10,kernel_size=(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        "../input/training_set/training_set",  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels


test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
        "../input/test_set/test_set",
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save_weights('first_try.h5')


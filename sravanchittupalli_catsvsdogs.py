# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import cv2

import random

import shutil

import zipfile
local_zip = '/kaggle/input/dogs-vs-cats/train.zip'

zip_ref = zipfile.ZipFile(local_zip , 'r')

zip_ref.extractall()

zip_ref.close()
train_images=[]

#test_images=[]

train_path = '/kaggle/working/train/'

#test_path = '/kaggle/working/test1/'

train_images_list = os.listdir(train_path)

#test_images_list = os.listdir(test_path)

print(train_images_list[: 10])
try:

    os.mkdir(os.path.join(train_path , 'cats'))

    os.mkdir(os.path.join(train_path , 'dogs'))

except FileExistsError :

    print("file already exists")

count_cats = 0

count_dogs = 0



for a in train_images_list:

    file_name = a.split('.')

    if(file_name[0] == 'cat'):

        shutil.move(os.path.join(train_path , a) , os.path.join(train_path , 'cats/cat{}.jpg'.format(count_cats)))

        count_cats = count_cats + 1 

    else:

        shutil.move(os.path.join(train_path , a) , os.path.join(train_path , 'dogs/dog{}.jpg'.format(count_dogs)))

        count_dogs = count_dogs + 1
cat_images = []

dog_images = []



cats_img_dir = os.path.join(train_path , 'cats')

dogs_img_dir = os.path.join(train_path , 'dogs')

cat_img_list = os.listdir(cats_img_dir)

dog_img_list = os.listdir(dogs_img_dir)





for a in cat_img_list:

    cat_images.append(os.path.join(cats_img_dir , a))

for a in dog_img_list:

    dog_images.append(os.path.join(dogs_img_dir , a))



sample = plt.imread(cat_images[0])

print(sample.shape)

plt.imshow(sample)

plt.show()
fig , x = plt.subplots(4,4)

for i in range (0 , 4 , 1):

    for j in range (0 , 4, 1):

        if(i==0 or i==1):

            x[i][j].imshow(plt.imread(cat_images[random.randint(0 , 12500)]))

        else:

            x[i][j].imshow(plt.imread(dog_images[random.randint(0 , 12500)]))
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1.0/255.0)



train_generator = train_datagen.flow_from_directory(

                                '/kaggle/working/train',

                                batch_size=1000,

                                class_mode='binary',

                                target_size=(150,150))
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.3),

    

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.3),

    

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.3),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1 , activation='sigmoid')

])
model.summary()
model.compile(optimizer='adam' , loss='binary_crossentropy', metrics=['acc'])
history = model.fit_generator(train_generator,

                             epochs=15,

                             steps_per_epoch=25,

                             verbose=2

                             )
model.save("rock-paper-scissor.h5")    
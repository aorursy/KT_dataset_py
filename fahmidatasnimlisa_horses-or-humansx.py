# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.metrics import classification_report, confusion_matrix



import matplotlib.pyplot as plt #for plotting things



from PIL import Image

import cv2

import tensorflow as tf
train_folder='/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/train/',

train_horse_dir=os.path.join('/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/train/horses/')

train_human_dir=os.path.join('/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/train/humans/')

validation_horse_dir=os.path.join('/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/validation/horses/')

validation_human_dir=os.path.join('/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/validation/humans/')
print('total training horse images:', len(os.listdir(train_horse_dir)))

print('total training human images:', len(os.listdir(train_human_dir)))

print('total validation horse images:', len(os.listdir(validation_horse_dir)))

print('total validation human images:', len(os.listdir(validation_human_dir)))
#horses pic

rand_horse=np.random.randint(0,len(os.listdir(train_horse_dir)))

horse_pic=os.listdir(train_horse_dir)[rand_horse]

print('Horse picture title:',horse_pic)



horses_pic_address='/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/train/horses/'+horse_pic



#human pic

rand_human=np.random.randint(0,len(os.listdir(train_human_dir)))

human_pic=os.listdir(train_human_dir)[rand_human]

print('Human picture title:',human_pic)



humans_pic_address='/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/train/humans/'+human_pic



#Load the images

horse_load = Image.open(horses_pic_address)

human_load = Image.open(humans_pic_address)



#Lets plot the images

f=plt.figure(figsize=(10,10))



a1=f.add_subplot(1,2,1)

img_plot=plt.imshow(horse_load)

a1.set_title('Horse')



a2=f.add_subplot(1,2,2)

img_plot=plt.imshow(human_load)

a2.set_title('Human')
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),#the input shape is the desired size of the image 300x300 with 3 bytes color

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),# Flatten the results to feed into a DNN

    tf.keras.layers.Dense(512,activation='relu'),# 512 neuron hidden layer

    tf.keras.layers.Dense(1,activation='sigmoid')# Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')

])
model.summary()
from tensorflow.keras.optimizers import RMSprop



model.compile(loss='binary_crossentropy',

             optimizer=RMSprop(lr=0.001),

             metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator



#augmentation on training, validation data



# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1/255)

validation_datagen = ImageDataGenerator(rescale=1/255)



# Flow training images in batches of 128 using train_datagen generator

train_batch = train_datagen.flow_from_directory(

                '/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/train/',# This is the source directory for training images

                target_size=(300,300),# All images will be resized to 300x300

                batch_size=128,

    # Since we use binary_crossentropy loss, we need binary labels

                class_mode='binary')



# Flow training images in batches of 32 using validation_datagen generator

validation_batch=validation_datagen.flow_from_directory(

                '/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human/validation/',# This is the source directory for validation images

                target_size=(300,300),# All images will be resized to 300x300

                batch_size=32,

    # Since we use binary_crossentropy loss, we need binary labels

                class_mode='binary')

hist=model.fit(

            train_batch,

            steps_per_epoch=8,

            epochs=15,

            verbose=1,

            validation_data=validation_batch,

            validation_steps=8)
#prediction=model.predict(validation_batch)

#print(prediction)
test_acc=model.evaluate_generator(validation_batch,steps=5)

print('the testing accuracy is:',test_acc[1]*100,'%')
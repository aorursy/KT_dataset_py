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

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import layers,models
# path of the cell-images-directory as starting point for the datagen.flow_from_directory() funktion

basis_path ="/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/"

# the pathes of the sub-directories

parasitized_path ="/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/"

uninfected_path ="/kaggle/input/cell-images-for-detecting-malaria/cell_images/Uninfected/"
%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

img = mpimg.imread('/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/C212ThinF_IMG_20151106_105941_cell_158.png')

#print(img)

imgplot = plt.imshow(img)

img = mpimg.imread('/kaggle/input/cell-images-for-detecting-malaria/cell_images/Parasitized/C136P97ThinF_IMG_20151005_140538_cell_96.png')

#print(img)

imgplot = plt.imshow(img)
# With keras ImageDataGenerator I generate batches of tensor image data with real-time data augmentation. 

# The data will be looped over (in batches).



from keras.preprocessing.image import ImageDataGenerator



# define the batchsize for the training

batch_size = 32

# set the image-size

(img_height,img_width) = (150,150)



#define an ImagaDataGenerator for data-augmentation for the images

datagen = ImageDataGenerator(

        rescale=1./255,

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest',

        validation_split=0.2 # important vor train-/validation splitting

)



test_datagen = ImageDataGenerator( # the test set is not pre-processed except for scaling

        rescale=1./255,

        validation_split=0.2 # test-dataset contains 50% of the dataset

)



# because of keras has now added train-/validation split from a single directory using ImageDataGenerator 

# I seperate my train-/validation-/testt sets by myself

# from keras-tutorial https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html





train_generator = datagen.flow_from_directory(

    basis_path,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary',

    subset='training') # set as training data



validation_generator = datagen.flow_from_directory(

    basis_path, # same directory as training data

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary',

    subset='validation') # set as validation data



test_generator = test_datagen.flow_from_directory(

    basis_path,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary',

    subset='validation') # only the subsets 'training' and 'validation' are known



def make_my_model ():

    model = tf.keras.Sequential(

    [

    tf.keras.layers.Reshape(input_shape=(3,img_height,img_width),target_shape=(img_height,img_width,3),name="image"),

    

    tf.keras.layers.Conv2D(filters=64, kernel_size=10, padding='same'), # no bias necessary before batch norm

    tf.keras.layers.BatchNormalization(), # no batch norm scaling necessary before "relu"

    tf.keras.layers.ReLU(), 

     

    tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same'), 

    tf.keras.layers.BatchNormalization(), 

    tf.keras.layers.ReLU(), 

     

    tf.keras.layers.MaxPool2D((2,2)), 

    tf.keras.layers.Dropout(0.2),

     

    tf.keras.layers.Conv2D(filters=128, kernel_size=5, padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.LeakyReLU(alpha=0.1), # in the inner layer, my results are better with LeakyRelu

    

    tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.LeakyReLU(alpha=0.1),  

     

    tf.keras.layers.MaxPool2D((2,2)),

    tf.keras.layers.Dropout(0.2), 

    

    tf.keras.layers.Flatten(),

     

    tf.keras.layers.Dense(256),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.LeakyReLU(alpha=0.1), 

        

    tf.keras.layers.Dense(128),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.LeakyReLU(alpha=0.1), 

     

    tf.keras.layers.Dropout(0.2),

     

    tf.keras.layers.Dense(32),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.LeakyReLU(alpha=0.1), 

    

    tf.keras.layers.Dense(1, activation='sigmoid')

       

    ])

    

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
model = make_my_model()



model.summary()
EPOCHS = 3

model.fit_generator(

    train_generator,

    steps_per_epoch = train_generator.samples // batch_size,

    validation_data = validation_generator, 

    validation_steps = validation_generator.samples // batch_size,

    epochs = EPOCHS)
scores = model.evaluate_generator(test_generator) 

print("Accuracy = ", scores[1])
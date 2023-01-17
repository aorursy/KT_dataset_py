# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from IPython.display import Image, display
from learntools.deep_learning.decode_predictions import decode_predictions
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, UpSampling2D, Lambda
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img
from tensorflow.keras import Input
import tensorflow as tf
import pylab as pl

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

image_size = 224
num_letters = 33
num_bgs = 2

li = []
letters1_data = pd.read_csv('/kaggle/input/classification-of-handwritten-letters/letters.csv').sample(frac=1).reset_index(drop=True)
#letters2_data = pd.read_csv('/kaggle/input/classification-of-handwritten-letters/letters2.csv')
#letters3_data = pd.read_csv('/kaggle/input/classification-of-handwritten-letters/letters3.csv')
li.append(letters1_data)
#li.append(letters2_data)
#li.append(letters3_data)
letters_data = pd.concat(li).sample(frac=1).reset_index(drop=True)

dict = {}
for i in letters_data.values:
    dict[i[1]] = i[0]

files=[]
for file in letters_data.file.values:
    files.append('/kaggle/input/classification-of-handwritten-letters/letters/' + file)

# plotting of fitting histories for neural networks
def history_plot(fit_history):
    pl.figure(figsize=(12,10)); pl.subplot(211)
    keys=list(fit_history.history.keys())[0:4]
    pl.plot(fit_history.history[keys[0]],
            color='slategray',label='train')
    pl.plot(fit_history.history[keys[2]],
            color='#4876ff',label='valid')
    pl.xlabel("Epochs"); pl.ylabel("Loss")
    pl.legend(); pl.grid()
    pl.title('Loss Function')     
    pl.subplot(212)
    pl.plot(fit_history.history[keys[1]],
            color='slategray',label='train')
    pl.plot(fit_history.history[keys[3]],
            color='#4876ff',label='valid')
    pl.xlabel("Epochs"); pl.ylabel("Accuracy")    
    pl.legend(); pl.grid()
    pl.title('Accuracy'); pl.show()

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width), interpolation="bicubic", color_mode="rgb") for img_path in img_paths]
    img_array = [img_to_array(img) for img in imgs]
    output = np.asarray(img_array) / 255
    return(output)

x = read_and_prep_images(files)
y = keras.utils.to_categorical(letters_data.label.values-1, num_letters)
z = keras.utils.to_categorical(letters_data.background.values, num_bgs)

(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.05)
(_, _, z_train, z_test) = train_test_split(x, z, test_size=0.05)

def MyUpSampling2D(size):
    return Lambda(lambda x: 
        tf.image.resize(x, size, method=tf.image.ResizeMethod.BICUBIC, antialias=True)
    )

def MyContrast():
    return Lambda(lambda x: 
        tf.image.adjust_contrast(x, 2.0)
    )

def MyGreyscale():
    return Lambda(lambda x: 
        tf.image.rgb_to_grayscale(x)
    )

resnet_model = ResNet50V2(include_top=False, weights='imagenet', layers=tf.keras.layers)
resnet_model.trainable = False

my_new_model = Sequential()

# upsample 32x32 images to meet resnets 224x224 resolution
my_new_model.add(MyUpSampling2D((224,224)))
my_new_model.add(MyContrast())
my_new_model.add(resnet_model)
my_new_model.add(Dense(256, activation='relu'))
my_new_model.add(Dropout(.25))
my_new_model.add(BatchNormalization())
my_new_model.add(Flatten())
my_new_model.add(Dense(num_letters, activation='softmax'))

my_new_model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
train_generator = ImageDataGenerator(rotation_range=0.25, zoom_range=0.25, shear_range=0.25)
train_generator.fit(x_train, augment=True)
train_generator = train_generator.flow(x_train, y_train, batch_size=100)

validation_generator = ImageDataGenerator()
validation_generator.fit(x_test)
validation_generator = validation_generator.flow(x_test, y_test, batch_size=50)
fit_stats = my_new_model.fit_generator(train_generator,
                                       steps_per_epoch=15,
                                       epochs=15,
                                       validation_data=validation_generator,
                                       validation_steps=1)
history_plot(fit_stats)
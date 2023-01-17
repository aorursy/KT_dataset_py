import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras # neural network models



from keras import optimizers

from keras.models import Model

from keras import applications

from keras.models import Sequential

from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Conv2D

from keras.layers import Dropout, Flatten, Dense

from keras.applications import InceptionV3, ResNet50, Xception, densenet



# For working with images

import cv2 as cv

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import tqdm



# Potentially useful tools - you do not have to use these

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical



import os



from numpy.random import seed

seed(42)

# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
# CONSTANTS

# You may not need all of these, and you may find it useful to set some extras



CATEGORIES = ['airplane','car','cat','dog','flower','fruit','motorbike','person']



IMG_WIDTH =  100

IMG_HEIGHT =  100

TRAIN_PATH = '../input/natural_images/natural_images/'

TEST_PATH = '../input/evaluate/evaluate/'



BATCH_SIZE = 256

LEARNING_RATE = 0.01

EPOCHS = 10
# To find data:

folders = os.listdir(TRAIN_PATH)



images = []



for folder in folders:

    files = os.listdir(TRAIN_PATH + folder)

    images += [(folder, file, folder + '/' + file) for file in files]



image_locs = pd.DataFrame(images, columns=('class','filename','file_loc'))



# data structure is three-column table

# first column is class, second column is filename, third column is image address relative to TRAIN_PATH

image_locs.head()
img_path = image_locs["file_loc"][0]

img_arr = mpimg.imread(TRAIN_PATH+img_path)



plt.imshow(img_arr)
def build_model():

    

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',

                     input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(64, (3, 3), padding='same'))

    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(512))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(len(CATEGORIES)))

    model.add(Activation('softmax'))

                

    model.compile(loss='categorical_crossentropy',

                      optimizer="RMSprop",

                      metrics=['categorical_accuracy'])



    return model
#train from all the data for the final predicition



#build our data generators

final_datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



final_generator = final_datagen.flow_from_directory(

    TRAIN_PATH,

    target_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=BATCH_SIZE,

    classes=CATEGORIES,

    class_mode='categorical')



final_model = build_model()



final_model.fit_generator(

    final_generator,

    steps_per_epoch = final_generator.samples // BATCH_SIZE,

    epochs = EPOCHS)
#save model



final_model.save("model.h5")
#predict images

#resize so it works with our trained model

#build our data generators

test_datagen = ImageDataGenerator(

    rescale=1./255)



test_generator = test_datagen.flow_from_directory(

    '../input/evaluate/',

    target_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=BATCH_SIZE,

    shuffle=False,

    classes=["evaluate"],

    class_mode='categorical')



predictions = final_model.predict_generator(test_generator, steps=1)

filenames = test_generator.filenames



remove_dir = lambda x: x.replace("evaluate/", "")

get_label = lambda x: CATEGORIES[np.argmax(x)]



filenames = list(map(remove_dir, filenames))

predictions = list(map(get_label, predictions))
# Save results



# results go in dataframe: first column is image filename, second column is category name

# category names are: airplane, car, cat, dog, flower, fruit, motorbike, person

df = pd.DataFrame()

df['filename'] = filenames

df['label'] = predictions

df = df.sort_values(by='filename')



print(df.head())



df.to_csv('results.csv', header=True, index=False)
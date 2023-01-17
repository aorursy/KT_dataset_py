import numpy as np

import pickle

import cv2

from os import listdir

from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation, Flatten, Dropout, Dense

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing import image

from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
EPOCHS = 25

INIT_LR = 1e-3

BS = 32

default_image_size = tuple((256, 256))

image_size = 0

directory_root = '/kaggle/input/-plant/_plant/'

width=256

height=256

depth=3
def convert_image_to_array(image_dir):

    try:

        image = cv2.imread(image_dir)

        if image is not None :

            image = cv2.resize(image, default_image_size)   

            return img_to_array(image)

        else :

            return np.array([])

    except Exception as e:

        print(f"Error : {e}")

        return None
image_list, label_list = [], []

try:

    print(" Loading images ...")

    root_dir = listdir(directory_root)



    for plant_folder in root_dir :

        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")

        

        for plant_disease_folder in plant_disease_folder_list:

            print(f"Processing {plant_disease_folder} ...")

            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")

                

            for single_plant_disease_image in plant_disease_image_list :

                if single_plant_disease_image == ".DS_Store" :

                    plant_disease_image_list.remove(single_plant_disease_image)



            for image in plant_disease_image_list[:200]:

                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"

                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:

                    image_list.append(convert_image_to_array(image_directory))

                    label_list.append(plant_disease_folder)

    print(" Completed")  

except Exception as e:

    print(f"Error : {e}")
image_size = len(image_list)

print(image_size)
label_binarizer = LabelBinarizer()

image_labels = label_binarizer.fit_transform(label_list)

print(label_binarizer.classes_)
np_image_list = np.array(image_list, dtype=np.float16) / 225.0
print(" Spliting images to train, test")

x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 
aug = ImageDataGenerator(

    rotation_range=25, width_shift_range=0.1,

    height_shift_range=0.1, shear_range=0.2, 

    zoom_range=0.2,horizontal_flip=True, 

    fill_mode="nearest")
disease_type = Sequential()

inputShape = (height, width, depth)

chanDim = -1

if K.image_data_format() == "channels_first":

    inputShape = (depth, height, width)

    chanDim = 1

disease_type.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape, activation = 'relu'))

disease_type.add(BatchNormalization(axis=chanDim))

disease_type.add(MaxPooling2D(pool_size=(2,2)))

disease_type.add(Dropout(0.3))





disease_type.add(Conv2D(32, (3, 3), padding="same",activation = 'relu'))

disease_type.add(BatchNormalization(axis=chanDim))

disease_type.add(MaxPooling2D(pool_size=(2, 2)))

disease_type.add(Dropout(0.3))



disease_type.add(Flatten())

disease_type.add(BatchNormalization())

disease_type.add(Dropout(0.3))

disease_type.add(Dense(activation="relu", units=128))

disease_type.add(Dense(activation="sigmoid", units=1))
disease_type.compile(loss="binary_crossentropy", optimizer='adam',metrics=["accuracy"])
history = disease_type.fit_generator(

    aug.flow(x_train, y_train, batch_size=BS),

    validation_data=(x_test, y_test),

    steps_per_epoch=len(x_train) // BS,

    epochs=EPOCHS

    )
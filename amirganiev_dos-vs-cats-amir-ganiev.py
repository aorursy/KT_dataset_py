#import needed libraries
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications.resnet50 import preprocess_input
#PREPARE DATA (Pictures)
#we will have two different ways to preprocess
#with ImageGenerator function and without, then we will compare the results
# NUMBER 1: my own function to preprocess data

img_size=224
def read_and_prep_images(img_paths, img_height=img_size, img_width=img_size):
    imgs= [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array=np.array([img_to_array(img) for img in imgs])
    output=preprocess_input(img_array)
    return(output)

imgs_paths_train1='../input/train'
imgs_paths_test1='../input/test'

image_data_train1=read_and_prep_images(imgs_paths_train1)
image_data_test1=read_and_prep_images(imgs_paths_test1)
#Number 2: Using ImageGenerator
#also allows us to use data_augmentation

image_size=224
imgs_paths_train2='../input/train'
imgs_paths_test2='../input/test'

data_generator_with_aug=ImageDataGenerator(preprocessing_function=preprocess_input,
                                           horizontal_flip=True,
                                           width_shift_range = 0.2,
                                           height_shift_range = 0.2)

data_generator_no_aug=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator_aug=data_generator_with_aug.flow_from_directory(
        directory='../input/train/',
        target_size=(image_size, image_size),
        batch_size=100,
        class_mode='categorical')

train_generator_no_aug=data_generator_no_aug.flow_from_directory(
        directory='../input/train/',
        target_size=(image_size, image_size),
        batch_size=100,
        class_mode='categorical')

test_generator=data_generator_no_aug.flow_from_directory(
        directory='../input/test/',
        target_size=(image_size, image_size),
        class_mode='categorical')
#CONSTRUCT A MODEL
#now image data is prepared with two different technics, lets see which one will work

num_classes=2
model=Sequential()
model.add(Conv2D(24, kernel_size=(3,3), strides=2,
                activation='relu',
                input_shape=(image_size, image_size,1)))
model.add(Dropout(0.5))
model.add(Conv2D(24, kernel_size=(3, 3), strides=2,
                activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(24, kernel_size=(3,3), strides=2,
                activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer='adam',
             metrics=['accuracy'])
#FINALLY RUN THE MODEL
#many different cases
#1. train with augmentation
model.fit(train_generator_aug,
                   epochs=3)
import keras
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
import time

NAME = 'board {}'.format(time.time())
tensorboard = TensorBoard(log_dir='../input/photos/{}'.format(NAME))

num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        '../input/photos/photos/train/',
        target_size=(image_size, image_size),
        batch_size=12,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
            '../input/photos/photos/val/',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs = 3,
        validation_data=validation_generator,
        validation_steps=1,
)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob as glob

p = 1
def find(path):
    plt.figure(figsize=(15,10))
    p = 1
    for im in glob.glob(path):
        plt.subplot(5,5,p)
        p = p + 1
        im = cv2.imread(im)
        im = cv2.resize(im,(224,224))
        x = preprocess_input(np.expand_dims(im.copy(), axis=0))
        if (my_new_model.predict_classes(x) == 0):
            txt = "Modiji"
        else :
            txt = "Rahul"
        plt.title(txt)
        plt.imshow(im)
find('../input/photos/photos/val/rural/*.jpg')
find('../input/photos/photos/val/urban//*.jpg')


% matplotlib inline

import h5py

import os

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import cv2

from keras.utils.io_utils import HDF5Matrix



import keras

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D, MaxPool2D, Flatten, Dense, Dropout, Activation, Input

from keras.optimizers import SGD

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer



from keras.applications import ResNet50

from keras.applications import InceptionV3

from keras.applications import Xception # TensorFlow ONLY

from keras.applications import VGG16, VGG19, InceptionV3

from keras.applications import imagenet_utils

from keras.applications.inception_v3 import preprocess_input

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping 

import warnings

warnings.filterwarnings('ignore')
models_filename = '../input/v8-vgg16-model-1/v8_vgg16_model_1.h5'

image_dir = '../input/food-101/food-101/food-101/images'

image_size = (224, 224)

batch_size = 16

epochs = 80
# 5gb of images won't fit in my memory. use datagenerator to go across all images.

train_datagen = ImageDataGenerator(

rescale = 1./255,

horizontal_flip = False,

fill_mode = "nearest",

zoom_range = 0,

width_shift_range = 0,

height_shift_range=0,

rotation_range=0)



train_generator = train_datagen.flow_from_directory(

image_dir,

target_size = (image_size[0], image_size[1]),

batch_size = batch_size, 

class_mode = "categorical")



num_of_classes = len(train_generator.class_indices)
model = VGG16(weights=None, include_top=False, input_shape=(image_size[0], image_size[1], 3))



#Adding custom Layers 

x = model.output

x = Flatten()(x)

x = Dense(101*2, activation="relu")(x)

x = Dense(101*2, activation="relu")(x)

predictions = Dense(101, activation="softmax")(x)

model_final = Model(input=model.input, output=predictions)

model_final.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

model_final.load_weights(models_filename)
preds = model_final.evaluate_generator(train_generator, steps=800, workers=8, use_multiprocessing=True)

preds
# routine for human evaluation - use the generator so we can see how well it can predict

for n in range(100):

    _ = train_generator.next()

    image, classifier = (_[0][0],_[1][0]) # take the first image from the batch

    index = np.argmax(classifier)

    answer = list(train_generator.class_indices.keys())[index]

    predicted = model_final.predict(np.asarray([image]))

    predicted_answer_index = np.argmax(predicted[0])

    predicted_answer = list(train_generator.class_indices.keys())[predicted_answer_index]



    plt.imshow(image)

    plt.show()



    print('correct answer is: ', answer)

    print()

    print('CNN thinks it''s:', predicted_answer)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import cv2
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet_v2 import ResNet152V2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, ConvLSTM2D, Conv3D, MaxPooling2D, Dropout, \
    MaxPooling3D, AveragePooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
import json

df = pd.read_csv('../input/fer2013/fer2013.csv')
def get_image(string_image):
    img = [int(i) for i in string_image.split()]
    return img

def load_data(df):
    train_x_array = []
    test_x_array = []
    
    for _, string_image in (df[df['Usage'] == 'Training']['pixels']).items():
        train_x_array.append(get_image(string_image))
    for _, string_image in (df[df['Usage'] == 'PublicTest']['pixels']).items():
        test_x_array.append(get_image(string_image))
    train_y = df[df['Usage'] == 'Training']['emotion'].to_numpy()
    test_y = df[df['Usage'] == 'PublicTest']['emotion'].to_numpy()
    
    train_x = np.array(train_x_array)
    test_x = np.array(test_x_array)
    
    return (train_x, test_x, train_y, test_y)
    
emotions = {0: 'Angry', 
            1: 'Disgust',
            2: 'Fear', 
            3: 'Happy', 
            4: 'Sad', 
            5: 'Surprise',  
            6: 'Neutral'}
train_x, test_x, train_y, test_y = load_data(df)

train_x = train_x.reshape(-1, 48, 48)
test_x = test_x.reshape(-1, 48, 48)

train_x = tf.image.resize(train_x, (224, 224))
test_x = tf.image.resize(test_x, (224, 224))

train_x = np.repeat(train_x[..., np.newaxis], 3, -1)
test_x = np.repeat(test_x[..., np.newaxis], 3, -1)

train_y = to_categorical(train_y, num_classes=len(emotions))
test_y = to_categorical(test_y, num_classes=len(emotions))


#train_x = tf.image.resize(train_x, (224, 224))
#test_x = tf.image.resize(test_x, (224, 224))
datagen =  ImageDataGenerator(rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.5,
        zoom_range=(0.9, 1.1),
        horizontal_flip=False,
        vertical_flip=False, 
        fill_mode='constant',
        cval=0)

datagen.fit(train_x)

train_x = train_x / 255.
test_x = test_x / 255.
model = ResNet152V2(weights='imagenet', include_top = False, input_shape = train_x[0].shape)

top_layer_model = model.output
top_layer_model = GlobalAveragePooling2D()(top_layer_model)
top_layer_model = Dense(1024, activation='relu')(top_layer_model)
top_layer_model = Dropout(0.5)(top_layer_model)
prediction_layer = Dense(output_dim=len(emotions.keys()), activation='softmax')(top_layer_model)

for layer in model.layers[:-20]:
    layer.trainable = False
    
model = Model(input=model.input, output=prediction_layer)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=64),
                    steps_per_epoch=len(train_x) / 64, epochs=20, validation_data=(test_x, test_y), callbacks = [EarlyStopping()], shuffle=True)
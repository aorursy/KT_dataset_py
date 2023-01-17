import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

import itertools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import random

import os,shutil

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from keras.layers.convolutional import Conv2D, MaxPooling2D,ZeroPadding2D

from __future__ import print_function



import warnings



from keras.models import Model

from keras.layers import Flatten, Dense, Input

from keras.layers import Convolution2D, MaxPooling2D

from keras.preprocessing import image



from keras import backend as K

K.set_image_dim_ordering('th')

K.set_image_data_format('channels_last')

from keras import layers

from keras import models

from keras import optimizers

from keras.layers import GlobalAveragePooling2D







TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels.h5'

src_path="../input"



print(os.listdir(src_path))



#constant value

VALID_SPIT=0.2

IMAGE_SIZE=80

BATCH_SIZE=20
label=[]

data=[]

counter=0

path="../input/train/train"

for file in os.listdir(path):

    image_data=cv2.imread(os.path.join(path,file), cv2.IMREAD_COLOR)

    image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))

    if file.startswith("cat"):

        label.append(0)

    elif file.startswith("dog"):

        label.append(1)

    try:

        data.append(image_data/255)

    except:

        label=label[:len(label)-1]

    counter+=1

    if counter%1000==0:

        print (counter," image data retreived")



data=np.array(data)

data=data.reshape((data.shape)[0],(data.shape)[1],(data.shape)[2],3)

label=np.array(label)

print (data.shape)

print (label.shape)

from sklearn.model_selection import train_test_split

train_data, valid_data, train_label, valid_label = train_test_split(

    data, label, test_size=0.2, random_state=42)

print(train_data.shape)

print(train_label.shape)

print(valid_data.shape)

print(valid_label.shape)
def VGG19(include_top=True, weights='imagenet',

          input_tensor=None):

   

    if weights not in {'imagenet', None}:

        raise ValueError('The `weights` argument should be either '

                         '`None` (random initialization) or `imagenet` '

                         '(pre-training on ImageNet).')

    # Determine proper input shape

    if K.image_dim_ordering() == 'th':

        if include_top:

            input_shape = (3, 224, 224)

        else:

            input_shape = (3, None, None)

    else:

        if include_top:

            input_shape = (80, 80, 3)

        else:

            input_shape = (None, None, 3)



    if input_tensor is None:

        img_input = Input(shape=input_shape)

    else:

        if not K.is_keras_tensor(input_tensor):

            img_input = Input(tensor=input_tensor)

        else:

            img_input = input_tensor

    # Block 1

    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)

    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)



    # Block 2

    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)

    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)



    # Block 3

    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)

    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)

    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)



    # Block 4

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)



    # Block 5

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    

    if include_top:

        x = Flatten(name='flatten')(x)

        x = Dense(256, activation='relu', name='fc2')(x)

        x = Dense(1, activation='sigmoid', name='predictions')(x)



    # Create model

    model = Model(img_input, x)

    if weights == 'imagenet':

        print('K.image_dim_ordering:', K.image_dim_ordering())

        if K.image_dim_ordering() == 'th':

            if include_top:

                weights_path = get_file('vgg19_weights_th_dim_ordering_th_kernels.h5',

                                        TH_WEIGHTS_PATH,

                                        cache_subdir='models')



    return model


model = VGG19(include_top=True, weights=None)

model.summary()
model.compile(loss='binary_crossentropy',optimizer=optimizers.adam(lr=1e-4),metrics=['acc'])





train_history=model.fit(train_data, train_label,validation_data=(valid_data,valid_label),epochs=20, batch_size=BATCH_SIZE)
import matplotlib.pyplot as plt

acc = train_history.history['acc']

val_acc = train_history.history['val_acc']

loss = train_history.history['loss']

val_loss = train_history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'blue', label='Training acc')

plt.plot(epochs, val_acc, 'red', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'blue', label='Training loss')

plt.plot(epochs, val_loss, 'red', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
from keras import Sequential

from keras.layers import *

import keras.optimizers as optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import *

import keras.backend as K

test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_directory("../input/test1",target_size=(80, 80),batch_size=32,class_mode='binary')
from tensorflow.python.keras.models import Sequential

from keras.models import load_model



print("-- Evaluate --")



scores = model.evaluate_generator(

            test_generator, 

            steps = 100)
Y_pred = model.predict(valid_data)

predicted_label=np.round(Y_pred,decimals=2)
import matplotlib.pyplot as plt

%matplotlib inline

from mlxtend.plotting import plot_confusion_matrix



# Get the confusion matrix



CM = confusion_matrix(valid_label, Y_pred.round())

fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(12, 12))

plt.xticks(range(2), ['Cat', 'Dog'], fontsize=16)

plt.yticks(range(2), ['Cat', 'Dog'], fontsize=16)

plt.show()
!pip install git+https://github.com/raghakot/keras-vis.git -U
import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.cm as cm

from vis.visualization import visualize_saliency, overlay, visualize_cam

from vis.utils import utils
correct_indices = []

incorrect_indices = []



# separate the correctly predicted results with incorrectly predicted ones

for i in range(len(Y_pred)):

    if Y_pred.round()[i] == valid_label[i]:

        correct_indices.append(valid_data[i])

    else:

        incorrect_indices.append(valid_data[i])



correct_indices = np.array(correct_indices)

incorrect_indices = np.array(incorrect_indices)



# print('-- Correctly Predicted Results\' Indices -
len(correct_indices)
len(incorrect_indices)
layer_idx = utils.find_layer_idx(model,'predictions')

penultimate_layer = utils.find_layer_idx(model,'block5_conv3')

model.layers[layer_idx].activation = activations.linear

model = utils.apply_modifications(model)



plt.rcParams['figure.figsize'] = (18, 6)

for i in range(-5, 5):



    f, ax = plt.subplots(1, 4)

    ax[0].set_title('original image')

    ax[0].imshow(valid_data[i])

  

    for j, modifier in enumerate([None, 'guided', 'relu']):

        grads = visualize_cam(model, layer_idx, filter_indices = 0, seed_input = valid_data[i], penultimate_layer_idx = penultimate_layer, backprop_modifier = modifier)

    

        if modifier is None:

            modifier = 'vanilla'



        ax[j + 1].set_title(modifier)

        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)

        ax[j + 1].imshow(overlay(jet_heatmap, valid_data[i]))
# Default imports present when loading a kaggle kernel



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
#import the relevant libs

import cv2

import matplotlib.pyplot as plt 

import seaborn as sns

import os

from PIL import Image

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

from keras.utils import np_utils



# Deep Learning - Keras - Model

import keras

from keras import models

from keras.models import Model

from keras.models import Sequential



# Deep Learning - Keras - Layers

from keras.layers import  Flatten

from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization

from keras.layers.pooling import _GlobalPooling1D

from keras.applications.nasnet import preprocess_input



# Deep Learning - Keras - Model Parameters and Evaluation Metrics

from keras import optimizers

from keras.optimizers import Adam 
parasitized_data = os.listdir('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/')

print(parasitized_data[:10]) #the output we get are the .png files



uninfected_data = os.listdir('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/')

print('\n')

print(uninfected_data[:10])
plt.figure(figsize = (12,12))

for i in range(6):

    plt.subplot(1, 6, i+1)

    img = cv2.imread('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized' + "/" + parasitized_data[i])

    plt.imshow(img)

    plt.title('PARASITIZED : 1')

    plt.tight_layout()

plt.show()
plt.figure(figsize = (12,12))

for i in range(6):

    plt.subplot(1, 6, i+1)

    img = cv2.imread('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected' + "/" + uninfected_data[i+1])

    plt.imshow(img)

    plt.title('UNINFECTED : 0')

    plt.tight_layout()

plt.show()
data = []

labels = []

for img in parasitized_data:

    try:

        img_read = plt.imread('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/' + "/" + img)

        img_resize = cv2.resize(img_read, (64,64))

        img_array = img_to_array(img_resize)

        data.append(img_array)

        labels.append(1)

    except:

        None

        

for img in uninfected_data:

    try:

        img_read = plt.imread('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected' + "/" + img)

        img_resize = cv2.resize(img_read, (64, 64))

        img_array = img_to_array(img_resize)

        data.append(img_array)

        labels.append(0)

    except:

        None
plt.imshow(data[0])

plt.show()
image_data = np.array(data)

labels = np.array(labels)
idx = np.arange(image_data.shape[0])

np.random.shuffle(idx)

image_data = image_data[idx]

labels = labels[idx]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(image_data, labels, test_size = 0.2, random_state = 101)
y_train = np_utils.to_categorical(y_train, num_classes = 2)

y_test = np_utils.to_categorical(y_test, num_classes = 2)
print(f'SHAPE OF TRAINING IMAGE DATA : {x_train.shape}')

print(f'SHAPE OF TESTING IMAGE DATA : {x_test.shape}')

print(f'SHAPE OF TRAINING LABELS : {y_train.shape}')

print(f'SHAPE OF TESTING LABELS : {y_test.shape}')
from keras.applications.vgg16 import VGG16

from keras.applications.vgg19 import VGG19

from keras.layers import Input
def buildModel(input_shape=(64, 64, 3), num_class=2):

    inputs = Input(input_shape)

    base_model = VGG16(include_top=False, input_shape=input_shape)

    #base_model = VGG19(include_top=False, input_shape=input_shape)

    x = base_model(inputs)



    output1 = GlobalMaxPooling2D()(x)

    output2 = GlobalAveragePooling2D()(x)

    output3 = Flatten()(x)



    outputs = Concatenate(axis=-1)([output1, output2, output3])



    outputs = Dropout(0.5)(outputs)

    outputs = BatchNormalization()(outputs)



    if num_class>1:

        outputs = Dense(num_class, activation="softmax")(outputs)

    else:

        outputs = Dense(1, activation="sigmoid")(outputs)



    model = Model(inputs, outputs)



    return model
input_shape = (64, 64, 3)

num_class = 2

model = buildModel(input_shape=input_shape, num_class=num_class)

model.summary()
input_shape = (64, 64, 3) # for VGG19

num_class = 2

model_19 = buildModel(input_shape=input_shape, num_class=num_class)

model_19.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
model_19.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
model_history_vgg16 = model.fit(x_train, y_train, epochs = 12, batch_size = 32)
h_19 = model_19.fit(x_train, y_train, epochs = 12, batch_size = 32)
plt.figure(figsize = (18,8))

plt.plot(range(12), model_history_vgg16.history['accuracy'], label = 'Training Accuracy')

plt.plot(range(12), model_history_vgg16.history['loss'], label = 'Taining Loss')

plt.xlabel("Number of Epoch's")

plt.ylabel('Accuracy/Loss Value')

plt.title('Training Accuracy and Training Loss')

plt.legend(loc = "best")
plt.figure(figsize = (18,8))# VGG19

plt.plot(range(12), h_19.history['accuracy'], label = 'Training Accuracy')

plt.plot(range(12), h_19.history['loss'], label = 'Taining Loss')

plt.xlabel("Number of Epoch's")

plt.ylabel('Accuracy/Loss Value')

plt.title('Training Accuracy and Training Loss')

plt.legend(loc = "best")
predictions = model.evaluate(x_test, y_test)
print(f'LOSS : {predictions[0]}')

print(f'ACCURACY : {predictions[1]}')
predictions_19 = model_19.evaluate(x_test, y_test)
print(f'LOSS : {predictions_19[0]}')#vgg19

print(f'ACCURACY : {predictions_19[1]}')
!nvidia-smi
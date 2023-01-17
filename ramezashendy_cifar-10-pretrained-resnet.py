# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras as k



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import numpy as np

import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid

from os import listdir, makedirs

from os.path import join, exists, expanduser

from tqdm import tqdm

from sklearn.metrics import log_loss, accuracy_score

from keras.preprocessing import image

from keras.applications.vgg16 import VGG16

from keras.applications.resnet50 import ResNet50

from keras.applications import xception

from keras.applications import inception_v3

from keras.applications.vgg16 import preprocess_input, decode_predictions

from sklearn.linear_model import LogisticRegression
#%%       Load and Test ResNet50

#----------------------------------------------

RESNET50_WEIGHTS = '../input/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

RESNET50_NOTOP_WEIGHTS = '../input/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# Load Keras' ResNet50 model that was pre-trained against the ImageNet database

model = k.applications.resnet50.ResNet50(weights=RESNET50_WEIGHTS)



# Load the image file, resizing it to 224x224 pixels (required by this model)

img = k.preprocessing.image.load_img("../input/Kuszma.JPG", target_size=(224, 224))



# Convert the image to a numpy array

x = k.preprocessing.image.img_to_array(img)



# Add a forth dimension since Keras expects a list of images

x = np.expand_dims(x, axis=0)
#%%               Processing

#----------------------------------------------



# Scale the input image to the range used in the trained network

x = k.applications.resnet50.preprocess_input(x)
#%%               Prediction

#----------------------------------------------

# Run the image through the deep neural network to make a prediction

predictions = model.predict(x)



# Look up the names of the predicted classes. Index zero is the results for the first image.

predicted_classes = k.applications.resnet50.decode_predictions(predictions, top=9)



print("This is an image of:")



for imagenet_id, name, likelihood in predicted_classes[0]:

    print(" - {}: {:2f} likelihood".format(name, likelihood))

# Load CIFAR10 data

(X_train, y_train), (_, _) = k.datasets.cifar10.load_data()
X_train.shape
np.unique(y_train)
fig, axes = plt.subplots(3,5, figsize=(16,10))



for i in range(3):

    for j in range(5):

        axes[i, j].imshow(X_train[np.random.randint(50000, size=1)[0]][:,:,:])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
X_train.shape
# X_train = X_train.reshape(X_train.shape[0], 30, 30, 3).astype('float32')

# X_test = X_test.reshape(X_test.shape[0], 3, 224, 224).astype('float32')
num_class = 10

# Convert class vectors to binary class matrices.

y_train = k.utils.to_categorical(y_train, num_class)

#y_test = k.utils.to_categorical(y_test, num_class)
pretrained_model = k.applications.resnet50.ResNet50(weights=RESNET50_WEIGHTS)
print('Output_layer_type= {}'.format(pretrained_model.layers[-1]))

print('Output_layer_shape= {}'.format(pretrained_model.layers[-1].output_shape))
pretrained_model.layers.pop()
print('Output_layer_type= {}'.format(pretrained_model.layers[-1]))

print('Output_layer_shape= {}'.format(pretrained_model.layers[-1].output_shape))
pretrained_model.layers
len(pretrained_model.layers)
for layer in pretrained_model.layers[0:-21]:

    layer.trainable = False
from keras.layers import ZeroPadding2D

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
model = k.models.Sequential()

model.add(ZeroPadding2D((96, 96), input_shape=(32, 32,  3)))

model.add(pretrained_model)

#model.add(k.layers.Flatten())

#model.add(k.layers.GlobalAveragePooling2D())

#model.add(k.layers.Dense(1024, activation='relu'))

model.add(k.layers.Dense(num_class, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print('Input Shape = {}'.format(model.layers[0].input_shape))

print('output Shape = {}'.format(model.layers[-1].output_shape))
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow(X_train, y_train, batch_size = 32)
test_set = test_datagen.flow(X_test, y_test, batch_size = 32)
import keras

keras.__version__
len_tr = len(training_set)

len_te = len(test_set)

print(len_tr)

print(len_te)
# model.fit_generator(training_set,

#                          samples_per_epoch = len_tr,

#                          nb_epoch = 25,

#                          validation_data = test_set,

#                          nb_val_samples = len_te)
model.fit_generator(training_set,

                         samples_per_epoch = len_tr,

                         nb_epoch = 300)
predicto = model.predict_classes(X_test)
predicto.shape
y_test.shape
from sklearn.metrics import accuracy_score

accuracy_score(y_test, predicto)
predicto
y_test
name = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
fig, axes = plt.subplots(3,5, figsize=(16,10))



for i in range(3):

    for j in range(5):

        r = np.random.randint(10000, size=1)[0]

        axes[i, j].imshow(X_test[r][:,:,:])

        #axes[i, j].title('dddd')

        print('this is a', name[y_test[r][0]], '-------- prediction is:', name[predicto[r]])
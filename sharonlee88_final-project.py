# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

#load data file         
data = pd.read_csv("/kaggle/input/blood-cells/dataset2-master/dataset2-master/labels.csv")
data
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16, preprocess_input

import matplotlib.pyplot as plt
%matplotlib inline

# load an image from file
image1 = load_img("/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TRAIN/MONOCYTE/_0_1173.jpeg", target_size=(168, 168))
#print of the image, this is a monocyte
plt.imshow(image1)
# load an image from file
image2 = load_img("/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TRAIN/EOSINOPHIL/_0_2142.jpeg", target_size=(168, 168))
#this is a eosinophil
plt.imshow(image2)
image3 = load_img("/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TRAIN/LYMPHOCYTE/_0_204.jpeg", target_size=(168, 168))
#lymphocyte
plt.imshow(image3)
image4 = load_img("/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TRAIN/NEUTROPHIL/_100_8981.jpeg", target_size=(168, 168))
#neutrophil
plt.imshow(image4)
#count how many images for each type of cells 
data['Category'].value_counts()
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import GaussianNoise
#in order to improve validation accuracyadded drop outs, noise
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(GaussianNoise(0.1))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(4, activation='softmax'))
    
    adam = Adam(lr=0.00001)
    model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
    return model

model= create_model()
model.summary()
train_gen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                               samplewise_center=False,  # set each sample mean to 0
                               featurewise_std_normalization=False,  # divide inputs by std of the dataset
                               samplewise_std_normalization=False,  
                               rescale = 1./255,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rotation_range = 10,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip = True)

test_gen = ImageDataGenerator(rescale = 1./255)
train = train_gen.flow_from_directory('/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TRAIN',
                                      target_size = (64, 64),
                                      batch_size = 16,
                                      class_mode = 'categorical')

test = test_gen.flow_from_directory('/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TEST',
                                    target_size = (64, 64),
                                    batch_size = 16,
                                    class_mode = 'categorical')
model.fit_generator(train,
                    steps_per_epoch = 500,
                    epochs = 100,
                    validation_data = test,
                    validation_steps = 500, shuffle=True)
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg16 import VGG16
#load model
model = VGG16()

#summarize the model
model.summary()
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dense
from itertools import islice
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import GlobalMaxPool2D, Dropout
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
# from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
# load model without classifier layers
model = VGG16(include_top=False, input_shape=(64, 64, 3))


for layer in model.layers:
  model.get_layer(layer.name).trainable = False

flat1 = GlobalMaxPool2D()(model.output)
class1 = Dense(1024, activation='relu')(flat1)
class2 = Dense(128, activation='relu')(class1)

z = Dense(128, activation='relu')(class2)
output = Dense(4, activation='softmax')(z)
# define new model
model = Model(inputs=model.inputs, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.00001), metrics=['accuracy'])
model.fit_generator(train,
                    steps_per_epoch = 500,
                    epochs = 30,
                    validation_data = test,
                    validation_steps = 200)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file = (os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.layers.normalization import BatchNormalization

from keras.applications.vgg19 import VGG19

image_size = [224, 224]
vgg = VGG19(input_shape = image_size + [3], weights = 'imagenet', include_top =  False)

from glob import glob

folders = glob('../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/*')

len(folders)


for layer in (vgg.layers):

    layer.trainable = False
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation = 'softmax')(x)
from keras import Model



model = Model(inputs = vgg.input, outputs = prediction)

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   fill_mode='nearest')



valid_datagen = ImageDataGenerator(rescale=1./255)



batch_size = 150

base_dir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"



training_set = train_datagen.flow_from_directory(base_dir+'/train',

                                                 target_size=(224, 224),

                                                 batch_size=batch_size,

                                                 class_mode='categorical')



valid_set = valid_datagen.flow_from_directory(base_dir+'/valid',

                                            target_size=(224, 224),

                                            batch_size=batch_size,

                                            class_mode='categorical')
device_name = tensorflow.test.gpu_device_name()

if "GPU" not in device_name:

    print("GPU device not found")

print('Found GPU at: {}'.format(device_name))
mod = model.fit_generator(

  training_set,

  validation_data= valid_set,

  epochs=10,

  steps_per_epoch=len(training_set),

  validation_steps=len(valid_set)

)
import matplotlib.pyplot as plt

plt.plot(mod.history['loss'], label='train loss')

plt.plot(mod.history['val_loss'], label='val loss')

plt.legend()

plt.show()

plt.savefig('LossVal_loss')



# plot the accuracy

plt.plot(mod.history['accuracy'], label='train acc')

plt.plot(mod.history['val_accuracy'], label='val acc')

plt.legend()

plt.show()

plt.savefig('AccVal_acc')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from kaggle_gcp import __file__

from keras.preprocessing import image

import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.style.use(['ggplot'])

import os

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

import json

from keras.models import model_from_json, load_model

import math

import random

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
fraction = 0.2

path ='../input/rockpaperscissors/rps-cv-images/paper'

files = os.listdir(path)

num_select = math.ceil(len(files) * fraction)

selected_files = random.sample(range(len(files)), num_select)

for i in selected_files:

    cmd = 'copy {}/{} p_test/{}'.format(path, files[i], files[i])

    os.system(cmd)

os.listdir('p_test')
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data_dir = '/kaggle/input/rockpaperscissors/rps-cv-images/'

all_gen = ImageDataGenerator(

    rescale=1./255,        

    horizontal_flip=True,

    height_shift_range=.2,

    width_shift_range=0.2,

    shear_range=0.1,

    zoom_range=0.2,

    vertical_flip = True,

    validation_split = 0.2

)  



train_gen = all_gen.flow_from_directory(

    data_dir,

    target_size = (200,200),

    batch_size=32,

    class_mode = 'categorical',

    shuffle=True,

    subset='training'

)



val_gen = all_gen.flow_from_directory(

    data_dir,

    target_size=(200,200),

    batch_size=32,

    class_mode='categorical',

    shuffle=False,

    subset = 'validation'

)
val_gen.save_to_dir()
model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 300x300 with 3 bytes color

    # This is the first convolution

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(300, activation='relu'),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(150, activation='relu'),

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')

    tf.keras.layers.Dense(3, activation='softmax')

])
model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer=RMSprop(lr=0.0005),

              metrics=['acc'])
history = model.fit_generator(

      train_gen,

      epochs=15,

      verbose=1,

      validation_data = val_gen)
os.chdir(r'/kaggle/working')

model.save('rps.h5')
from IPython.display import FileLink

FileLink(r"rps.h5")
# Option 1: Save Weights + Architecture

model_test.save_weights('model_weights.h5')

with open('model_architecture.json', 'w') as f:

    f.write(model_test.to_json())
from IPython.display import FileLink

FileLink(r"model_architecture.json")
model_test = tf.keras.models.load_model('rps.h5')
history = model_test.fit_generator(

      train_gen,

      epochs=15,

      verbose=1,

      validation_data = val_gen)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()

plt.show()
uploaded =  __file__.upload()



for fn in uploaded.keys():

  # predicting images

  path = fn

  img = image.load_img(path, target_size=(200, 200))

  x = image.img_to_array(img)

  x = np.expand_dims(x, axis=0)



  images = np.vstack([x])

  classes = model.predict(images, batch_size=32)

  print(fn)

  print(classes)
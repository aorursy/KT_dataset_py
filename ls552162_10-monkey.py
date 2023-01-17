# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from tensorflow import keras

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

%matplotlib inline

print(tf.__version__)
train_dir = '/kaggle/input/10-monkey-species/training/training'

valid_dir  = '/kaggle/input/10-monkey-species/validation/validation'

label_dir = '/kaggle/input/10-monkey-species/monkey_labels.txt'



print(os.path.exists(train_dir))

print(os.path.exists(valid_dir))

print(os.path.exists(label_dir))

labels = pd.read_csv(label_dir,header=0)

print(labels)
width = 128

height = 128

chanels = 3

batch_size = 64

num_class = 10

train_datagen = keras.preprocessing.image.ImageDataGenerator(

											rescale = 1./255,

  											 rotation_range = 40,

   											 width_shift_range = 0.2,

    										height_shift_range = 0.2,

 										   shear_range = 0.2,

  											 horizontal_flip = True,

  											 fill_mode = 'nearest'

				)



valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dir,

    target_size = (width,height),

    batch_size = batch_size,

    seed = 7,

    shuffle = True,

    class_mode = 'categorical')

valid_generator = valid_datagen.flow_from_directory(valid_dir,

                                                    target_size = (width,height),

                                                    batch_size = batch_size,

                                                    seed = 7,

                                                    shuffle = False,

                                                    class_mode = 'categorical')

train_num = train_generator.samples

valid_num = valid_generator.samples

print(train_num,valid_num)
for i in range(2):

    x,y = train_generator.next()

    print(x.shape,y.shape)

    print(y)
model = keras.models.Sequential()



model.add(keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',

                              input_shape=(width,height,chanels)))

          

model.add(keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))



          

model.add(keras.layers.MaxPool2D(pool_size=2))



model.add(keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))

model.add(keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))

model.add(keras.layers.MaxPool2D(pool_size=2))      

model.add(keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))

model.add(keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))

model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128,activation='relu'))

model.add(keras.layers.Dense(10,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',

             optimizer='adam',

             metrics=['acc'])
epochs = 300

history = model.fit_generator(train_generator,

                              epochs= epochs,

                             steps_per_epoch = train_num//batch_size,

                             validation_data = valid_generator,

                             validation_steps = valid_num//batch_size)
print(history.history.keys())
def plot_learning_curves(history,label,epcohs,min_value,max_value):

    data = {}

    data[label] = history.history[label]

    data['val_'+label] = history.history['val_'+label]

    pd.DataFrame(data).plot(figsize=(8,5))

    plt.grid(True)

    plt.axis([0,epochs,min_value,max_value])

    plt.show()

plot_learning_curves(history,'acc',epochs,0,1)

plot_learning_curves(history,'loss',epochs,0,2.5)
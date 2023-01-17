from tensorflow import keras

from tensorflow.keras import layers, models, optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # making graphics



# Input data files are available in the "../input/" directory.

import os

train_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/'

os.listdir(train_dir)



# Running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_datagen = ImageDataGenerator(rescale=1./255, 

                                   validation_split = 1001/5216)



train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size = (150, 150),

    batch_size=64,

    class_mode='binary',

    subset = "training")



dev_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size = (150, 150),

    batch_size=64,

    class_mode='binary',

    subset = "validation")
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation = 'relu'))

model.add(layers.Dense(1, activation = 'sigmoid'))



model.compile(loss = 'binary_crossentropy',

              optimizer = optimizers.RMSprop(lr = 1e-4),

              metrics = ['acc'])



model.summary()
history = model.fit_generator(

      train_generator,

      steps_per_epoch = None, # = 3365//16

      verbose = 1,

      epochs = 10,

      validation_data = dev_generator,

      validation_steps = None # = 1441//16

      )



# Save the model

model.save('case_2_run_1.h5')
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(acc))
import matplotlib.pyplot as plt



plt.plot(epochs, acc, 'bo-', label='Training acc')

plt.plot(epochs, val_acc, 'r*-', label='Validation acc')

plt.title('Training and validation accuracy')

plt.grid()

plt.legend()



plt.figure()

plt.plot(epochs, loss, 'bo-', label='Training loss')

plt.plot(epochs, val_loss, 'r*-', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.grid()



plt.show()
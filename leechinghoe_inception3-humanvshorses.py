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

        print(os.path.join(dirname, filename))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install Keras==2.4.2

import tensorflow as tf

from tensorflow.keras import Model
train_horses_dir="../input/horses-or-humans-dataset/horse-or-human/train/horses"

train_humans_dir="../input/horses-or-humans-dataset/horse-or-human/train/humans"

validation_horses_dir="../input/horses-or-humans-dataset/horse-or-human/validation/horses"

validation_humans_dir="../input/horses-or-humans-dataset/horse-or-human/validation/humans"



train_horses_fnames = os.listdir(train_horses_dir)

train_humans_fnames = os.listdir(train_humans_dir)

validation_horses_fnames = os.listdir(validation_horses_dir)

validation_humans_fnames = os.listdir(validation_humans_dir)



print(len(train_horses_fnames))

print(len(train_humans_fnames))

print(len(validation_horses_fnames))

print(len(validation_humans_fnames))



from tensorflow.keras import layers
train_dir="../input/horses-or-humans-dataset/horse-or-human/train"

validation_dir="../input/horses-or-humans-dataset/horse-or-human/validation"
#Augmentation

train_datagen = ImageDataGenerator(rescale=1./255,

                                  rotation_range=40,

                                  width_shift_range=0.2,

                                  height_shift_range=0.2,

                                  zoom_range=0.2,

                                  shear_range=0.2,

                                  horizontal_flip=True,

                                  fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,

                                                   batch_size=20,

                                                    class_mode='binary',

                                                   target_size=(150,150))



validation_generator =  test_datagen.flow_from_directory( validation_dir,

                                                   batch_size=20,

                                                    class_mode='binary',

                                                   target_size=(150,150))
#set up transfer learning

from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model=InceptionV3(input_shape=(150,150,3), include_top=False,weights='imagenet')
for layer in pre_trained_model.layers:

    layer.trainable=False

  

pre_trained_model.summary()
last_layer=pre_trained_model.get_layer('mixed7')

last_output=last_layer.output

x = layers.Flatten()(last_output)

x = layers.Dense(1024, activation='relu')(x)

x = layers.Dropout(0.2)(x)                  

x = layers.Dense  (1, activation='sigmoid')(x)           



model = Model(pre_trained_model.input, x) 



model.compile(optimizer = 'adam', 

              loss = 'binary_crossentropy', 

              metrics = ['acc'])



model.summary()
history = model.fit(train_generator,

            validation_data = validation_generator,

            steps_per_epoch = 52,

            epochs = 10,

            validation_steps = 25,

            verbose = 1,use_multiprocessing = False)
import matplotlib.pyplot as plt

plt.plot(history.history['acc'],'b')

plt.plot(history.history['val_acc'],'r')

plt.show()
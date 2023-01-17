# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

# Data                

# └───train

# │   └───daisy

# │   │      daisy_images.jpeg     

# │   └───dandelioin

# │   │      dandelion_images.jpeg

# │   └───rose

# │   │      rose_images.jpeg     

# │   └───sunflower

# │   │      sunflower_images.jpeg

# │   └───tulip

# │         tulip_images.jpeg

# │

# └───Validation

# │   └───daisy

# │   │      daisy_images.jpeg     

# │   └───dandelioin

# │   │      dandelion_images.jpeg

# │   └───rose

# │   │      rose_images.jpeg     

# │   └───sunflower

# │   │      sunflower_images.jpeg

# │   └───tulip

#            tulip_images.jpeg
import os

import shutil

from os.path import isfile, join, abspath, exists, isdir, expanduser

from os import listdir, makedirs, getcwd, remove

from pathlib import Path
import pandas as pd

import numpy as np
# Check for the directory and if it doesn't exist, make one.

cache_dir = expanduser(join('~', '.keras'))

if not exists(cache_dir):

    makedirs(cache_dir)

    

# make the models sub-directory

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)

# original dataset folder, you can see above

input_path = Path('/kaggle/input/flowers-recognition/flowers')

flowers_path = input_path / 'flowers'
flowers = []

flower_types = os.listdir(flowers_path)

for species in flower_types:

    all_flowers = os.listdir(flowers_path / species)

    for flower in all_flowers:

        flowers.append((species, str(flowers_path/species) \

        + '/' + flower))

flowers = pd.DataFrame(data=flowers, 

                       columns = ['category', 'image'],

                       index = None

                      )
flowers.head()
# Let's check how many samples for each category are present

print("Total number of flowers in the dataset: ", len(flowers))

fl_count = flowers['category'].value_counts()

print("Flowers in each category: ")

print(fl_count)
# Make a parent directory `data` and two sub directories `train` and `valid`

%mkdir -p data/train

%mkdir -p data/valid



# Inside the train and validation sub=directories, make sub-directories for each catgeory

%cd data

%mkdir -p train/daisy

%mkdir -p train/tulip

%mkdir -p train/sunflower

%mkdir -p train/rose

%mkdir -p train/dandelion



%mkdir -p valid/daisy

%mkdir -p valid/tulip

%mkdir -p valid/sunflower

%mkdir -p valid/rose

%mkdir -p valid/dandelion



%cd ..
for category in fl_count.index:

    samples = flowers['image'][flowers['category'] == category].values

    perm = np.random.permutation(samples)

    # Copy first 100 samples to the validation directory and rest to the train directory

    for i in range(100):

        name = perm[i].split('/')[-1]

        shutil.copyfile(perm[i],'./data/valid/' + str(category) + '/'+ name)

    for i in range(101,len(perm)):

        name = perm[i].split('/')[-1]

        shutil.copyfile(perm[i],'./data/train/' + str(category) + '/' + name)
import numpy as np

import pandas as pd

import seaborn as sns

sns.set_style('darkgrid')

import matplotlib.pyplot as plt

import matplotlib.image as mimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D

from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.utils import plot_model



import tensorflow as tf

print("Tensorflow version:", tf.__version__)
# Define the generators

batch_size = 32

img_size = 240

# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(

        rescale=1./255,

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(rescale=1./255)





train_generator = train_datagen.flow_from_directory("data/train/",

                                                    target_size=(img_size,img_size),

                                                    batch_size=batch_size,

                                                    class_mode='categorical',

                                                    shuffle=True)



validation_generator = test_datagen.flow_from_directory("data/valid/",

                                                    target_size=(img_size,img_size),

                                                    batch_size=batch_size,

                                                    class_mode='categorical',

                                                    shuffle=False)
# SHOWING AUGUEMENTED iMAGES

from keras.preprocessing import image

fnames = [os.path.join('data/train/rose', fname) for

fname in os.listdir('data/train/rose')]

img_path = fnames[1]

img = image.load_img(img_path, target_size=(240, 240))



x = image.img_to_array(img)

x = x.reshape((1,) + x.shape)

i = 0

f, axes = plt.subplots(1,4,figsize=(14,4))

for batch in train_datagen.flow(x, batch_size=1):

    imgplot = axes[i].imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0:

        break

plt.show()
steps_per_epoch = train_generator.n//train_generator.batch_size

validation_steps = validation_generator.n//validation_generator.batch_size



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,

                              patience=2, min_lr=0.00001, mode='auto')

checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy',

                             save_weights_only=True, mode='max', verbose=1)

callbacks = [checkpoint, reduce_lr]
# Initialising the CNN

model = Sequential()



# 1 - Convolution

model.add(Conv2D(64,(3,3), padding='same', input_shape=(240, 240,3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# 2nd Convolution layer

model.add(Conv2D(128,(5,5), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# 3rd Convolution layer

model.add(Conv2D(512,(3,3), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# 4th Convolution layer

model.add(Conv2D(512,(3,3), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# Flattening

model.add(Flatten())



# Fully connected layer 1st layer

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))



# Fully connected layer 2nd layer

model.add(Dense(512))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))



model.add(Dense(5, activation='softmax'))



opt = Adam(lr=0.0005)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
epochs_cnn = 25

history_cnn = model.fit(

    x=train_generator,

    steps_per_epoch=steps_per_epoch,

    epochs=epochs_cnn,

    validation_data = validation_generator,

    validation_steps = validation_steps,

    callbacks=callbacks

)
acc = history_cnn.history['accuracy']

val_acc = history_cnn.history['val_accuracy']

loss = history_cnn.history['loss']

val_loss = history_cnn.history['val_loss']

epochs = range(1, len(acc) + 1)



f, axes = plt.subplots(1,2,figsize=(14,4))



axes[0].plot(epochs, acc, 'b', label='Training acc')

axes[0].plot(epochs, val_acc, 'r', label='Validation acc')

axes[0].legend()



axes[1].plot(epochs, loss, 'b', label='Training loss')

axes[1].plot(epochs, val_loss, 'r', label='Validation loss')

axes[1].legend()



plt.show()
from keras.applications import VGG19
pre_trained_model = VGG19(input_shape=(224,224,3), include_top=False, weights="imagenet")



for layer in pre_trained_model.layers[:19]:

    layer.trainable = False



model_vgg = Sequential([

    pre_trained_model,

    MaxPooling2D((2,2) , strides = 2),

    Flatten(),

    Dense(5 , activation='softmax')])

model_vgg.compile(optimizer = "adam" , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model_vgg.summary()
epochs_vgg = 25

history_vgg = model.fit(

    x=train_generator,

    steps_per_epoch=steps_per_epoch,

    epochs=epochs_cnn,

    validation_data = validation_generator,

    validation_steps = validation_steps,

    callbacks=callbacks

)
acc = history_vgg.history['accuracy']

val_acc = history_vgg.history['val_accuracy']

loss = history_vgg.history['loss']

val_loss = history_vgg.history['val_loss']

epochs = range(1, len(acc) + 1)



f, axes = plt.subplots(1,2,figsize=(14,4))



axes[0].plot(epochs, acc, 'b', label='Training acc')

axes[0].plot(epochs, val_acc, 'r', label='Validation acc')

axes[0].legend()



axes[1].plot(epochs, loss, 'b', label='Training loss')

axes[1].plot(epochs, val_loss, 'r', label='Validation loss')

axes[1].legend()



plt.show()
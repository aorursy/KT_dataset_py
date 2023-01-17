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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow.keras import models, layers, optimizers, regularizers

from tensorflow.keras.preprocessing import image

import os

import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import LearningRateScheduler

import math



## data augmentation is done using ImageDataGenerator instance



base_dir = '../input/seismic-classification-data/Seismic_data'



## create train, validation, test dir



train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'validation')

test_dir = os.path.join(base_dir, 'test')



## create cat and dog dir in all three



train_anticline_dir = os.path.join(train_dir, 'anticline')

train_fault_dir = os.path.join(train_dir, 'fault')



validation_anticline_dir = os.path.join(validation_dir, 'anticline')

validation_fault_dir = os.path.join(validation_dir, 'fault')



test_anticline_dir = os.path.join(test_dir, 'anticline')

test_fault_dir = os.path.join(test_dir, 'fault')





def model_plots(history):

    history_dict = history.history



    # plot histories



    epochs = history.epoch

    epochs = epochs[1:]

    epochs.append(len(epochs)+1)

    

    ## training loss and acc

    acc = history_dict['acc']

    loss = history_dict['loss']

    

    ## validation loss and acc

    val_acc = history_dict['val_acc']

    val_loss = history_dict['val_loss']

    

    plt.figure("Losses")

    plt.plot(epochs, loss, 'bo', label='Training loss')

    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.legend()

    plt.title("Training and Validation loss")

    plt.xlabel("Epochs")

    plt.ylabel("Losses")

    plt.show()

    

    

    plt.figure("Accuracy")

    plt.plot(epochs, acc, 'bo', label='Training accuracy')

    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

    plt.legend()

    plt.title("Training and Validation accuracy")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.show()









# # Vizualize some images

# datagen = image.ImageDataGenerator(

#     rotation_range=40,

#     width_shift_range=0.2,

#     height_shift_range=0.2,

#     shear_range=0.2,

#     zoom_range=0.2,

#     horizontal_flip=True,

#     fill_mode='nearest'

#     )

# fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]



# img_path = fnames[3]

# img = image.load_img(img_path, target_size=(150,150))

# x = image.img_to_array(img)

# x = x.reshape((1,) + x.shape)



# i=0

# for batch in datagen.flow(x, batch_size=1):

#     plt.figure(i)

#     imgplot = plt.imshow(image.array_to_img(batch[0]))

#     i = i+1

#     if i%4==0:

#         break

# plt.show()

x_dim = 200

y_dim = 200



model = models.Sequential()

model.add(layers.Conv2D(64, (2,2), activation='relu', input_shape=(x_dim,y_dim,1)))

model.add(layers.Conv2D(64, (2,2), activation='relu'))

model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128, (2,2), activation='relu'))

model.add(layers.Conv2D(128, (2,2), activation='relu'))

model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(256, (2,2), activation='relu'))

model.add(layers.Conv2D(256, (2,2), activation='relu'))

model.add(layers.MaxPooling2D(4,4))

#model.add(layers.Conv2D(512, (3,3), activation='relu'))

#model.add(layers.Conv2D(512, (3,3), activation='relu'))

#model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu'))

model.add(layers.Dense(356, activation='relu'))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(4,activation='sigmoid'))



model.compile(

    optimizer = optimizers.RMSprop(learning_rate=0.01),

    loss = 'binary_crossentropy',

    metrics=['acc']

    )



train_datagen = image.ImageDataGenerator(

    rescale=1./255,

    rotation_range=5,

    horizontal_flip=True,

    width_shift_range=0.2,

    height_shift_range=0.2,

    #shear_range=0.2,

    fill_mode="nearest"

    )

test_datagen = image.ImageDataGenerator(rescale=1./255)



## generators are also called loaders e.g. train_dl

train_gen = train_datagen.flow_from_directory(

    directory=train_dir,

    target_size=(x_dim,y_dim),

    batch_size=20,

    seed = 96,

    color_mode='grayscale',

    class_mode='categorical'

    )

validation_gen = test_datagen.flow_from_directory(

    directory=validation_dir,

    target_size=(x_dim,y_dim),

    batch_size=20,

    seed = 96,

    color_mode='grayscale',

    class_mode='categorical'

    )

tf.autograph.experimental.do_not_convert

def schedule(epoch):

    lr = 2e-6

    if epoch>15:

        lr = 2e-5

    if epoch>25:

        lr = 1e-4

    if epoch>45:

        lr = 5e-5

    if epoch>55:

        lr = 2e-5

    if epoch>80:

        lr = 3e-6

    return lr

callbacks_list = [LearningRateScheduler(schedule)]

history = model.fit_generator(

    train_gen,

    steps_per_epoch=len(train_gen),

    epochs=100,

    validation_data=validation_gen,

    validation_steps=len(validation_gen),

    callbacks=callbacks_list,

    verbose=2

    )



model_plots(history)



model.save('seismic_bin.h5')
test_gen = test_datagen.flow_from_directory(

    directory=test_dir,

    target_size=(x_dim,y_dim),

    color_mode='grayscale',

    class_mode='categorical'

    )

model.evaluate_generator(test_gen)
import matplotlib.pyplot as plt

from itertools import compress 

image_number = 2

classes = list(train_gen.class_indices.keys())

prediction = list(model.predict_proba(test_gen[0][0])[image_number]>0.4)

res = list(compress(classes, prediction))

print(res)

plt.imshow(test_gen[0][0][image_number].reshape(200,200))
model.predict_proba(test_gen[0][0])*100>40
plt.imshow(test_gen[0][0][2][:,:,0])
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from pathlib import Path

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline



root = '../input'



train_dir = Path(f'{root}/training/training/')

test_dir = Path(f'{root}/validation/validation/')
#label info

cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']

label_df = pd.read_csv(f"{root}/monkey_labels.txt", names=cols, skiprows=1)

label_df
labels = label_df['Common Name']

labels
def image_show(num_image,label):

    from matplotlib import pyplot as plt

    import random

    import cv2

    import os

    for i in range(num_image):

        imgdir = Path(f'{root}/training/training/' + label)

        imgfile = random.choice(os.listdir(imgdir))

        img = cv2.imread(f'{root}/training/training/'+ label +'/'+ imgfile)

        plt.figure(i)

        plt.imshow(img)

        plt.title(imgfile)

    plt.show()
print(labels[4])

image_show(3,'n4')
from keras.preprocessing.image import ImageDataGenerator



height = 150

width = 150

batch_size = 64

seed = 100



# Training generator

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')



train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(height, width),

    batch_size=batch_size,

    seed=seed,

    shuffle=True,

    class_mode='categorical')



# Test generator

test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=(height, width),

    batch_size=batch_size,

    seed=seed,

    shuffle=False,

    class_mode='categorical')



train_num = train_generator.samples

validation_num = validation_generator.samples 
def get_net(num_classes):

    from keras.models import Sequential

    from keras.layers import Conv2D, Activation, BatchNormalization, GlobalAvgPool2D, MaxPooling2D, Dropout



    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), strides=2))

    model.add(Activation('relu'))



    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), strides=2))

    model.add(Activation('relu'))



    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), strides=2))

    model.add(Activation('relu'))



    model.add(Conv2D(512, (1, 1), strides=2))

    model.add(Activation('relu'))

    model.add(Conv2D(num_classes, (1, 1)))

    model.add(GlobalAvgPool2D())

    model.add(Activation('softmax'))

    return model



num_classes = 10

net = get_net(num_classes)

net.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['acc'])

net.summary()
from keras.callbacks import ModelCheckpoint, EarlyStopping

filepath=("monkey.h5f")

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# = EarlyStopping(monitor='val_acc', patience=15)

callbacks_list = [checkpoint]#, stopper]



epochs = 200



history = net.fit_generator(train_generator,

                              steps_per_epoch= train_num // batch_size,

                              epochs=epochs,

                              validation_data=train_generator,

                              validation_steps= validation_num // batch_size,

                              callbacks=callbacks_list, 

                              verbose = 1

                             )
def visualized_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)



    plt.title('Training and validation accuracy')

    plt.plot(epochs, acc, 'red', label='Training acc')

    plt.plot(epochs, val_acc, 'blue', label='Validation acc')

    plt.legend()



    plt.figure()

    plt.title('Training and validation loss')

    plt.plot(epochs, loss, 'red', label='Training loss')

    plt.plot(epochs, val_loss, 'blue', label='Validation loss')

    plt.legend()

    

    plt.show()

    

visualized_history(history)
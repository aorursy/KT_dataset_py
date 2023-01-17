# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import os

os.environ['CUDA_VISIBLE_DEVICES']='1'



from keras.applications import Xception

from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

from keras.models import Model

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.regularizers import l2



CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',

              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

# split validation set

split=True



if split:

    import random

    import shutil

    # make dir and mv

    os.mkdir('./dev/')

    for category in CATEGORIES:

        os.mkdir('./dev/' + category)

        name = os.listdir('./train/' + category)

        random.shuffle(name)

        todev = name[:int(len(name) * .2)]

        for file in todev:

            shutil.move(os.path.join('train', category, file), os.path.join('dev', category))



# lr decay schedule

def lr_schedule(epoch):

    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120epochs.

    Called automatically every epoch as part of callbacks during training.

    # Arguments

        epoch (int): The number of epochs

    # Returns

        lr (float32): learning rate

    """

    lr = 1e-4

    if epoch > 120:

        lr *= 1e-2

    elif epoch > 80:

        lr *= 1e-1

    print('Learning rate: ', lr)

    return lr
# data generator 

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    rotation_range=50,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=True)



train_generator = train_datagen.flow_from_directory(

    './train',

    target_size=(299, 299),

    batch_size=16,

    class_mode='categorical',

    shuffle=True)



val_datagen = ImageDataGenerator(rescale=1. / 255)



val_generator = val_datagen.flow_from_directory(

    './dev',

    target_size=(299, 299),

    batch_size=16,

    class_mode='categorical',

    shuffle=True)
# pretrain dense layer

# to avoid large gradient to destroy the pretrained model

# build model

tensorboard = TenserBoard('./logs')



basic_model = Xception(include_top=False, weights='imagenet', pooling='avg')



for layer in basic_model.layers:

    layer.trainable = False



input_tensor = basic_model.input

# build top

x = basic_model.output

x = Dropout(.5)(x)

x = Dense(len(CATEGORIES), activation='softmax')(x)



model = Model(inputs=input_tensor, outputs=x)

model.compile(optimizer=RMSprop(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])



model.fit_generator(train_generator, epochs=40, 

                    validation_data=val_generator,

                    callbacks=[tensorboard],

                    workers=4,

                    verbose=0)
# train with whole model

# train model

for layer in model.layers:

    layer.W_regularizer = l2(1e-2)

    layer.trainable = True



model.compile(optimizer=RMSprop(lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])



# call backs

checkpointer = ModelCheckpoint(filepath='./checkpoint/weights_xception.h5', verbose=1,

                               save_best_only=True)





lr = LearningRateScheduler(lr_schedule)



# train dense layer

model.fit_generator(train_generator, 

                    steps_per_epoch=400,

                    epochs=150, 

                    validation_data=val_generator,

                    callbacks=[checkpointer, tensorboard, lr],

                    initial_epoch=40,

                    workers=4,

                    verbose=0)





model.save('xception.h5')
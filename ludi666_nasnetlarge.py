#ls /kaggle/input/dogs-vs-cats-redux-kernels-edition/train 
#ls /kaggle/input/dogs-vs-cats-redux-kernels-edition/test
!mkdir /kaggle/output

!mkdir /kaggle/output/train 

!mkdir /kaggle/output/train/cat

!mkdir /kaggle/output/train/dog
import shutil

import os
train_dir = '/kaggle/output/train'

train_dir_cat = os.path.join(train_dir, 'cat')

train_dir_dog = os.path.join(train_dir, 'dog')

for dirname, _, filenames in os.walk('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train'):

    for filename in filenames:

        if 'cat.' in filename:

            shutil.copy(os.path.join(dirname, filename), os.path.join(dirname, train_dir_cat))

        if 'dog.' in filename:

            shutil.copy(os.path.join(dirname, filename), os.path.join(dirname, train_dir_dog))
#!ls /kaggle/output/train/cat
import keras

from keras.preprocessing.image import ImageDataGenerator
# Data Generator

train_generator = ImageDataGenerator(

    rotation_range=30.0, 

    width_shift_range=0.1,

    height_shift_range=0.1, 

    brightness_range=None, 

    shear_range=0.1,

    zoom_range=0.2, 

    fill_mode='nearest', 

    horizontal_flip=True, 

    vertical_flip=True,

    validation_split=0.1

)
train_iter = train_generator.flow_from_directory(

    train_dir,

    classes=['cat', 'dog'], 

    class_mode='categorical', 

    target_size=(331, 331),

    batch_size=64,

    subset='training'

)

valid_iter = train_generator.flow_from_directory(

    train_dir,

    classes=['cat', 'dog'], 

    class_mode='categorical', 

    target_size=(331, 331),

    batch_size=64,

    subset='validation'

)
from keras.applications import NASNetLarge, ResNet50

from keras.layers import Dense, GlobalAveragePooling2D

from keras.models import Model
objective='categorical_crossentropy'

metrics=['accuracy']

optimizer='adam'

base_model = ResNet50(

    weights="imagenet",

    include_top=False,

    pooling=None

)

for layer in base_model.layers:

    layer.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation='relu')(x)

x = Dense(64, activation='relu')(x)

predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=2, verbose=0, mode='auto', cooldown=0, min_lr=0)

early_stop = EarlyStopping(monitor='val_acc', patience=3, verbose=0, mode='auto')

history = model.fit_generator(

    train_iter,

    steps_per_epoch=len(train_iter),

    validation_data=valid_iter,

    validation_steps=len(valid_iter),

    epochs=20,

    verbose=1,

    shuffle=True,

    callbacks=[reduce_lr, early_stop]

)
from PIL import Image

import numpy as np

import pandas as pd

ID = []

Prediction = []



for dirname, _, filenames in os.walk('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test'):

    for filename in filenames:

        # print('Processing: ',os.path.join(dirname, filename))

        idx = filename.split('.jpg')[0]

        ID.append(idx)

        img = Image.open(os.path.join(dirname, filename))

        img = img.resize((331,331))

        img = img.convert('RGB')

        img = np.array(img)

        img = img[np.newaxis, :, :, :]

        pred = model.predict(img)[0][1]

        Prediction.append(pred)

        
print(len(ID), len(Prediction))
Result = pd.DataFrame({'id':ID,'label':Prediction})
Result.to_csv('submission_NASNetLarge.csv',index=False)
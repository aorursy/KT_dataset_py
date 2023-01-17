# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

import cv2

from tqdm import tqdm, tqdm_notebook

import matplotlib.pyplot as plt



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.applications.resnet50 import ResNet50

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



seed = 4529

np.random.seed(seed)
base_dir = os.path.join("..", "input") # set base directory

train_df = pd.read_csv(os.path.join(base_dir, "train.csv"))

train_dir = os.path.join(base_dir, "train/train")

test_dir = os.path.join(base_dir, "test/test")



# print(os.listdir(train_dir))

print(train_df.head())
%load_ext tensorboard.notebook

%tensorboard --logdir logs
# train_images = []

# train_labels = []

# images = train_df['id'].values



# for image_id in tqdm_notebook(images):

#     image = np.array(cv2.imread(train_dir + "/" + image_id))

#     train_images.append(image)

    

#     label = train_df[train_df['id'] == image_id]['has_cactus'].values[0]

#     train_labels.append(label)

    

# train_images = np.asarray(train_images)

# train_images = train_images / 255.0

# train_labels = np.asarray(train_labels)



# print("Number of Training images: " + str(len(train_images)))
train_df['has_cactus'] = train_df['has_cactus'].astype(str)



batch_size = 64

train_size = 15750

validation_size = 1750



datagen = ImageDataGenerator(

    rescale=1./255,

    horizontal_flip=True,

    vertical_flip=False,

    brightness_range=(1, 1.3),

    shear_range=0.05,

    validation_split=0.1)



data_args = {

    "dataframe": train_df,

    "directory": train_dir,

    "x_col": 'id',

    "y_col": 'has_cactus',

    "shuffle": True,

    "target_size": (32, 32),

    "batch_size": batch_size,

    "class_mode": 'binary'

}



train_generator = datagen.flow_from_dataframe(**data_args, subset='training')

validation_generator = datagen.flow_from_dataframe(**data_args, subset='validation')
model = Sequential([

    Conv2D(64, (3,3), padding='same', activation="relu", input_shape=(32, 32, 3)),

    BatchNormalization(),

    Conv2D(64, (3,3), padding='same', activation="relu"),

    BatchNormalization(),

    Conv2D(64, (3,3), padding='same', activation="relu"),

    BatchNormalization(),

    MaxPooling2D(2,2),

    Dropout(0.4),

    Conv2D(128, (3,3), padding='same', activation="relu"),

    BatchNormalization(),

    Conv2D(128, (3,3), padding='same', activation="relu"),

    BatchNormalization(),

    Conv2D(128, (3,3), padding='same', activation="relu"),

    BatchNormalization(),

    MaxPooling2D(2,2),

    Dropout(0.4),

    Conv2D(256, (3,3), padding='same', activation="relu"),

    BatchNormalization(),

    Conv2D(256, (3,3), padding='same', activation="relu"),

    BatchNormalization(),

    MaxPooling2D(2,2),

    Dropout(0.4),

    Conv2D(256, (3,3), padding='same', activation="relu"),

    BatchNormalization(),

    Conv2D(256, (3,3), padding='same', activation="relu"),

    BatchNormalization(),

    MaxPooling2D(2,2),

    Dropout(0.4),

    GlobalAveragePooling2D(),

    Dense(units=256, activation='relu'),

    Dropout(0.5),

    Dense(units=256, activation='relu'),

    Dropout(0.5),

    Dense(units=1, activation='sigmoid')

])



model.compile(optimizer=Adam(lr=0.001), 

                 loss='binary_crossentropy',

                 metrics=['acc'])

model.summary()
ckpt_path = 'aerial_cactus_detection.hdf5'



earlystop = EarlyStopping(monitor='val_acc', patience=10, verbose=1, restore_best_weights=True)

reducelr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, min_lr=1.e-6)

modelckpt_cb = ModelCheckpoint(ckpt_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

tb = TensorBoard()



callbacks = [earlystop, reducelr, modelckpt_cb, tb]
history = model.fit_generator(train_generator,

              validation_data=validation_generator,

              steps_per_epoch=train_size//batch_size,

              validation_steps=validation_size//batch_size,

              epochs=100, verbose=1, 

              shuffle=True,

              callbacks=callbacks)
# Training plots

epochs = [i for i in range(1, len(history.history['loss'])+1)]



plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")

plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")

plt.legend(loc='best')

plt.title('loss')

plt.xlabel('epoch')

plt.show()



plt.plot(epochs, history.history['acc'], color='blue', label="training_accuracy")

plt.plot(epochs, history.history['val_acc'], color='red',label="validation_accuracy")

plt.legend(loc='best')

plt.title('accuracy')

plt.xlabel('epoch')

plt.show()
test_df = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))

print(test_df.head())

test_images = []

images = test_df['id'].values



for image_id in images:

    test_images.append(cv2.imread(os.path.join(test_dir, image_id)))

    

test_images = np.asarray(test_images)

test_images = test_images / 255.0

print("Number of Test set images: " + str(len(test_images)))
pred = model.predict(test_images)

test_df['has_cactus'] = pred

test_df.to_csv('aerial-cactus-submission.csv', index = False)
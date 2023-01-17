import os

import zipfile

import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score

import tensorflow as tf

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.optimizers import Adam
import tensorflow
tensorflow.__version__
# <SPECIFY YOUR PATH HERE>

path = "../input/kaggledays-china/"

os.listdir(path) # you should have test and train data here
# This initial size of the pictures - change it, if you're going to crop them manually

img_width, img_height = 100, 100



# Specify here where your data are located - I use combined, 3-channel data



train_data_dir = path + 'train3c/train3c' # train data - it should have directories for each class inside

test_data_dir = path + 'test3c'  # test data - you have to keep 2-level directory structure here





nb_train_samples = 5024

nb_validation_samples = 1257

epochs = 20

batch_size = 512
if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)
# I use sklearn's implementation of ROC AUC, out of convenience

def auroc(y_true, y_pred):

    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
# Feel free to add more layers or neurons if you have enough computing power

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))



model.add(Flatten())

model.add(Dense(32))

model.add(Activation('relu'))

model.add(Dropout(0.20))

model.add(Dense(1))

model.add(Activation('sigmoid'))
# I star with quite small lr 

optimizer = Adam(lr=0.0003)

model.compile(loss='binary_crossentropy',

              optimizer=optimizer,

              metrics=[auroc])
# this is the augmentation configuration we will use for training

# you can try using more transformations here, by uncommenting lines below



train_datagen = ImageDataGenerator(

    rescale=1. / 255,

#     shear_range=0.2,

#     zoom_range=[0.5,1.0],

#     brightness_range=[0.2,1.0],

#     rotation_range=90,

#     horizontal_flip=True

)
# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(

    rescale=1. / 255

)



train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary',

    seed = 42

)



# avoid shuffling her - it'll make hard keeping your predictions and labels together

test_generator = test_datagen.flow_from_directory(

    test_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode=None,

    seed = 42,

    shuffle = False

)
model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=epochs,

    #you can add a generator with validation data here, to monitor your training process in each epoch

)
pred=model.predict_generator(test_generator,

verbose=1)

pred[:10] # check if format is correct
pred = [x[0] for x in pred]
labels = (test_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())
# Save the data to required sumbission format



filenames=test_generator.filenames

results=pd.DataFrame({"id":[x.split("/")[-1].split(".")[0] for x in filenames],

                      "is_star":pred})

results.to_csv("results.csv",index=False)
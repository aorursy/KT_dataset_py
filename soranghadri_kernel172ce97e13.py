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
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import pickle
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras import applications
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint, History
from keras.layers import Dense, Input, UpSampling2D, Conv2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras_tqdm import TQDMCallback, TQDMNotebookCallback
img_width, img_height = 28, 28
train_data_dir = "../input/fruits/fruits-360_dataset/fruits-360/Training"
validation_data_dir = "../input/fruits/fruits-360_dataset/fruits-360/Test"
batch_size = 64
nb_epoch = 5
nb_channels= 3
img = plt.imread('../input/fruits/fruits-360_dataset/fruits-360/Training/Banana/66_100.jpg')
plt.imshow(img)
encoding_dim = 256
def flattened_generator(generator):
    for batch in generator:
        yield (batch.reshape(-1,img_width*img_height*nb_channels), batch.reshape(-1,img_width*img_height*nb_channels))
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None, shuffle=True)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None, shuffle=True)
def AE_FC():

    # this is our input layer
    input_img = Input(shape=(img_height*img_width*nb_channels,))
    
    # this is the bottleneck vector
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    
    # this is the decoded layer, with the same shape as the input
    decoded = Dense(img_height*img_width*nb_channels, activation='sigmoid')(encoded)
    
    return Model(input_img, decoded)
autoencoder = AE_FC()
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
checkpoint = ModelCheckpoint(filepath = "model_weights_ae_fc.h5", save_best_only=True,monitor="val_loss", mode="min" )
history = History()
autoencoder.fit_generator(
    flattened_generator(train_generator),
    samples_per_epoch=math.floor(41322  / batch_size),
    nb_epoch=nb_epoch,
    validation_data=flattened_generator(validation_generator),
    nb_val_samples=math.floor(13877  / batch_size),
    verbose=0,
    callbacks=[history, checkpoint, TQDMNotebookCallback(leave_inner=True, leave_outer=True)])

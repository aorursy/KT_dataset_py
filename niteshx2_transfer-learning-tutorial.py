import os

from os import listdir, makedirs

from os.path import join, exists, expanduser



from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.models import Sequential, Model

from keras.layers import Dense, GlobalAveragePooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K

import tensorflow as tf

# Any results you write to the current directory are saved as output.
from subprocess import check_output

print(check_output(["ls", "../input/dataset/dataset/"]).decode("utf8"))

# dimensions of our images.

img_width, img_height = 224, 224 # we set the img_width and img_height according to the pretrained models we are

# going to use. The input size for ResNet-50 is 224 by 224 by 3.



#train_data_dir = '../input/fruits/fruits-360_dataset_2018_06_03/fruits-360/Training/'

data_dir = '../input/dataset/dataset/'

#validation_data_dir = '../input/fruits/fruits-360_dataset_2018_06_03/fruits-360/Validation/'

#nb_train_samples = 30000

#nb_validation_samples = 10000

batch_size = 16
# Will generate augmented images : image augmentation is curcial part of training deep learning networks



train_datagen = ImageDataGenerator(

    #featurewise_center=True, 

    samplewise_center=False,

    #featurewise_std_normalization=True,

    samplewise_std_normalization=False,

    zca_whitening=False,

    #zca_epsilon=1e-06,

    rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    brightness_range=[0.05 , 0.1],

    shear_range=0.2,

    zoom_range=0.2,

    channel_shift_range=0.1,

    fill_mode='nearest',

    cval=0.0,

    horizontal_flip=True,

    validation_split=0.3)





#test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(

    data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical',

    subset = 'training',

    interpolation = 'nearest',

    #cval = 0,

)



validation_generator = train_datagen.flow_from_directory(

    data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical',

    subset = 'validation',

   interpolation = 'nearest',

    #cval = 0,

)
import pandas as pd

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)
training_data = pd.DataFrame(train_generator.classes, columns=['classes'])

testing_data = pd.DataFrame(validation_generator.classes, columns=['classes'])
def create_stack_bar_data(col, df):

    aggregated = df[col].value_counts().sort_index()

    x_values = aggregated.index.tolist()

    y_values = aggregated.values.tolist()

    return x_values, y_values
x1, y1 = create_stack_bar_data('classes', training_data)

x1 = list(train_generator.class_indices.keys())



trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name="Class Count")

layout = dict(height=400, width=1200, title='Class Distribution in Training Data', legend=dict(orientation="h"), 

                yaxis = dict(title = 'Class Count'))

fig = go.Figure(data=[trace1], layout=layout);

iplot(fig);
x1, y1 = create_stack_bar_data('classes', testing_data)

x1 = list(validation_generator.class_indices.keys())



trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name="Class Count")

layout = dict(height=400, width=1100, title='Class Distribution in Validation Data', legend=dict(orientation="h"), 

                yaxis = dict(title = 'Class Count'))

fig = go.Figure(data=[trace1], layout=layout);

iplot(fig);
### 







'''#import inception with pre-trained weights. do not include fully #connected layers

inception_base = applications.InceptionResNetV2(weights='imagenet', include_top=False)



# add a global spatial average pooling layer

x = inception_base.output

x = GlobalAveragePooling2D()(x)

# add a fully-connected layer

x = Dense(512, activation='relu')(x)

# and a fully connected output/classification layer

predictions = Dense(13, activation='softmax')(x)

# create the full network so we can train on it

inception_transfer = Model(inputs=inception_base.input, outputs=predictions)'''
#import inception with pre-trained weights. do not include fully #connected layers

inception_base_vanilla = applications.DenseNet201(weights='imagenet', include_top=False)



# add a global spatial average pooling layer

x = inception_base_vanilla.output

x = GlobalAveragePooling2D()(x)

# add a fully-connected layer

x = Dense(2048, activation='relu')(x)

#x = Dense(128, activation='relu')(x)

# and a fully connected output/classification layer

predictions = Dense(13, activation='softmax')(x)

# create the full network so we can train on it

inception_transfer_vanilla = Model(inputs=inception_base_vanilla.input, outputs=predictions)
'''inception_transfer.compile(loss='categorical_crossentropy',

              optimizer=optimizers.Adam(lr=1e-4),

              metrics=['accuracy'])

'''

inception_transfer_vanilla.compile(loss='categorical_crossentropy',

              optimizer=optimizers.SGD(lr=1e-4),

              metrics=['accuracy'])
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
import keras

batch_size = 16



nb_train_samples = 15000

nb_validation_samples = 5960

import numpy as np

steps_per_epoch = np.ceil(nb_train_samples / batch_size)

val_steps = np.ceil(nb_validation_samples / batch_size)



pld = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_lr= 1e-8)

import tensorflow as tf

with tf.device("/device:GPU:0"):

    history_vanilla = inception_transfer_vanilla.fit_generator(train_generator,

    epochs=50, shuffle = True, verbose = 1, validation_data = validation_generator,

    validation_steps = val_steps , steps_per_epoch = steps_per_epoch , use_multiprocessing = True)
'''with tf.device("/device:GPU:0"):

    history_vanilla = inception_transfer_vanilla.fit_generator(train_generator,

    epochs=12, shuffle = True, verbose = 1, validation_data = validation_generator,

    validation_steps = val_steps , steps_per_epoch = steps_per_epoch , use_multiprocessing = True)'''
import matplotlib.pyplot as plt

# summarize history for accuracy

#plt.plot(history_pretrained.history['val_acc'])

plt.plot(history_vanilla.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['Pretrained'], loc='upper left')

plt.show()

# summarize history for loss

#plt.plot(history_pretrained.history['val_loss'])

plt.plot(history_vanilla.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['Pretrained'], loc='upper left')

plt.show()
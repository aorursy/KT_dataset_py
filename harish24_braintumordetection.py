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
!pip install split_folders
!pip install tensorflow==2.1.0
import split_folders

split_folders.ratio('../input/brain-mri-images-for-brain-tumor-detection/brain_tumor_dataset',
                    output='brain-tumor',
                    seed=1337,
                    ratio=(.8, .1,.1))
import tensorflow as tf

train_folder = '../working/brain-tumor/train'

validate_folder = '../working/brain-tumor/val'
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rotation_range=30,
                               zoom_range=20,
                               horizontal_flip=True,
                               rescale=1. / 255)
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
train_data_gen = train_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_folder,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT,
                                                            IMG_WIDTH),
                                               class_mode='binary',
                                               color_mode='grayscale')
val_gen = ImageDataGenerator(rotation_range=30,
                             zoom_range=20,
                             horizontal_flip=True,
                             rescale=1. / 255)
val_data_gen = val_gen.flow_from_directory(batch_size=batch_size,
                                           directory=validate_folder,
                                           shuffle=True,
                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                           class_mode='binary',
                                           color_mode='grayscale')
from tensorflow.keras.layers import Dense, Dropout, Convolution2D, MaxPool2D, Flatten, InputLayer, Conv2D
from tensorflow.keras.models import Sequential

model = Sequential([
    Conv2D(32, (3, 3),
           padding='same',
           strides=2,
           activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPool2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPool2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPool2D((2, 2), strides=2),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPool2D((3, 3), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
%load_ext tensorboard
import os
import datetime

logdir = os.path.join("logs",
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
r = model.fit_generator(train_data_gen,
                        epochs=12,
                        validation_data=val_data_gen,
                        callbacks = [tensorboard_callback])
%tensorboard --logdir logs
print(r.history.keys())

r.history
!pip install plotly
import plotly.express as px
import plotly.graph_objs as go

data = [go.Scatter(x = r.epoch, y = r.history['loss'], mode='lines',name='loss')]

layout = go.Layout(title = 'Learning Curve',
                        xaxis = dict(title = 'epochs'),
                        yaxis = dict(title = ''))

fig = go.Figure(data = data)

fig.add_trace(go.Scatter(x=r.epoch,y=r.history['accuracy'],mode='lines',name='accuracy'))

fig.add_trace(go.Scatter(x=r.epoch,y=r.history['val_loss'],mode='lines',name='val_loss'))

fig.add_trace(go.Scatter(x=r.epoch,y=r.history['val_accuracy'],mode='lines',name='val_accuracy'))

fig.update_layout(title='Learning Curve',
                   xaxis_title='Epochs',
                   yaxis_title='Metrics and Loss',
                 template = 'plotly_dark')

fig.show()


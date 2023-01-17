# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import Input
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file = pd.read_csv("../input/lego-minifigures-classification/index.csv")
file["class_id"] = file["class_id"].astype(str)
file["path"] = "../input/lego-minifigures-classification/" + file["path"]
    

train_data = file[file["train-valid"]=="train"].reset_index()

test_data = file[file["train-valid"]=="valid"].reset_index()

print(train_data.path[0])
file.groupby("class_id")["path"].count()
gen = ImageDataGenerator(rescale=1./255, rotation_range=90,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

val_gen= ImageDataGenerator(rescale=1./255)

val_gen = val_gen.flow_from_dataframe(test_data,
    x_col="path",
    y_col="class_id",
    weight_col=None,
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    seed = 44,
    interpolation="nearest",
    validate_filenames=False)


train_gen = gen.flow_from_dataframe(
    train_data,
    x_col="path",
    y_col="class_id",
    weight_col=None,
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed = 44,
    interpolation="nearest",
    validate_filenames=False)

from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
include_top=False,
input_shape=(256, 256, 3))

conv_base.trainable = False
con = 64*16

input_tensor = Input(shape=(256,256,3))
x = conv_base(input_tensor)
x = layers.SeparableConv2D(con,(3,3,), activation='relu')(x)
x = layers.SeparableConv2D(con,(3,3,), activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.SeparableConv2D(con,(3,3,), activation='relu')(x)
x = layers.MaxPool2D(2,2)(x)
x = layers.Dropout(0.2)(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation = 'relu')(x)
x = layers.BatchNormalization()(x)
output_tensor = layers.Dense(9, activation='softmax')(x)
model = models.Model(input_tensor, output_tensor)
model.compile(optimizer = 'Adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
callbacks_list = [
keras.callbacks.EarlyStopping(
monitor='acc',
patience=1,
),
keras.callbacks.ModelCheckpoint(
filepath='my_model.h5',
monitor='val_loss',
save_best_only=True,),
    
keras.callbacks.ReduceLROnPlateau(
monitor='val_loss',
factor=0.5,
patience=3,
)
]

epochs = 80
history = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch = 2, validation_data=val_gen, callbacks= callbacks_list, verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
my_model = keras.models.load_model('./my_model.h5')
test_loss, test_acc = my_model.evaluate_generator(val_gen)

print(test_loss, test_acc)
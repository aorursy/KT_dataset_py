# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats
import cv2
import glob
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.layers import Dense
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv('../input/landmark-recognition-2020/train.csv')
train_df.info()
print(len(train_df))
train_list = glob.glob('../input/landmark-recognition-2020/train/*/*/*/*')
example = cv2.imread(train_list[10000])
print(example.shape)
plt.figure(figsize=(20,10))
plt.imshow(example)
print(example.shape)
train_df["filename"] = train_df.id.str[0]+"/"+train_df.id.str[1]+"/"+train_df.id.str[2]+"/"+train_df.id+".jpg"
train_df["label"] = train_df.landmark_id.astype(str)
train_df.head()
from collections import Counter

c = train_df.landmark_id.values
count = Counter(c).most_common(1000)
print(len(count), count[-1])
# only keep 1000 classes
keep_labels = [i[0] for i in count]
train_keep = train_df[train_df.landmark_id.isin(keep_labels)]
val_rate = 0.2
batch_size = 32
gen = ImageDataGenerator(validation_split=val_rate)

train_gen = gen.flow_from_dataframe(
    train_keep,
    directory="/kaggle/input/landmark-recognition-2020/train/",
    x_col="filename",
    y_col="label",
    weight_col=None,
    target_size=(256, 256),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    subset="training",
    interpolation="nearest",
    validate_filenames=False)
    
val_gen = gen.flow_from_dataframe(
    train_keep,
    directory="/kaggle/input/landmark-recognition-2020/train/",
    x_col="filename",
    y_col="label",
    weight_col=None,
    target_size=(256, 256),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    subset="validation",
    interpolation="nearest",
    validate_filenames=False)
#AlexNet
#model = tf.keras.Sequential([
#    keras.layers.Conv2D(64, 8, activation="relu", padding="same",
#                        input_shape=[256, 256, 3]),
#    keras.layers.Conv2D(96, 11, strides=4, activation="relu", padding="valid"),
#    keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="valid"),
#    keras.layers.Conv2D(256, 5, strides=1, activation="relu", padding="same"),
#    keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="valid"),
#    keras.layers.Conv2D(384, 3, strides=1, activation="relu", padding="same"),
#    keras.layers.Conv2D(384, 3, strides=1, activation="relu", padding="same"),
#    keras.layers.Conv2D(256, 3, strides=1, activation="relu", padding="same"),
#    keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="valid"),
#    keras.layers.Flatten(),
#    keras.layers.Dense(4096, activation="relu"),
#    keras.layers.Dropout(0.5),
#    keras.layers.Dense(4096, activation="relu"),
#    keras.layers.Dropout(0.5),
#    keras.layers.Dense(1000, activation='softmax')
#])
#ResNet-34
#class ResidualUnit(keras.layers.Layer):
#    def __init__(self, filters, strides=1, activation="relu", **kwargs):
#        super().__init__(**kwargs)
#        self.activation = keras.activations.get(activation)
#        self.main_layers = [
#            keras.layers.Conv2D(filters, 3, strides=strides,
#                                padding="same", use_bias=False),
#            keras.layers.BatchNormalization(),
#            self.activation,
#            keras.layers.Conv2D(filters, 3, strides=1,
#                                padding="same", use_bias=False),
#            keras.layers.BatchNormalization()]
#        self.skip_layers = []
#        if strides>1:
#            self.skip_layers = [
#                keras.layers.Conv2D(filters, 1, strides=strides,
#                                    padding="same", use_bias=False),
#                keras.layers.BatchNormalization()]
#    
#    def call(self, inputs):
#        Z = inputs
#        for layer in self.main_layers:
#            Z = layer(Z)
#        skip_Z = inputs
#        for layer in self.skip_layers:
#            skip_Z = layer(skip_Z)
#        return self.activation(Z + skip_Z)
#model = keras.models.Sequential()
#model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[224,224,3],
#                             padding="same", use_bias=False))
#model.add(keras.layers.BatchNormalization())
#model.add(keras.layers.Activation("relu"))
#model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
#prev_filters=64
#for filters in [64]*3 + [128]*4 + [256]*6 + [512]*3:
#    strides=1 if filters==prev_filters else 2
#    model.add(ResidualUnit(filters, strides=strides))
#    prev_filters = filters
#model.add(keras.layers.GlobalAvgPool2D())
#model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(1000, activation="softmax"))
model = tf.keras.Sequential([
    tf.keras.applications.EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(256,256,3)
    ),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1000, activation='softmax')
])

#model = tf.keras.Sequential([
#    efn.EfficientNetB3(
#        input_shape=(256, 256, 3),
#        weights='imagenet',
#        include_top=False
#    ),
#    keras.layers.GlobalAveragePooling2D(),
#    keras.layers.Dense(1000, activation='softmax')
#])

model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics=['categorical_accuracy']
)

epochs = 1
train_steps = int(len(train_keep)*(1-val_rate))//batch_size
val_steps = int(len(train_keep)*val_rate)//batch_size

model_checkpoint = ModelCheckpoint("model_efnB3.h5", save_best_only=True, verbose=1)
history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs,validation_data=val_gen,
                              validation_steps=val_steps, callbacks=[model_checkpoint])

model.save("model.h5")
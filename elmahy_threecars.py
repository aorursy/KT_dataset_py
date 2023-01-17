!mkdir cars
!mkdir cars/lexus
!mkdir cars/toyota
!mkdir cars/hyundai


!find ../input/commoncars/Car\ Model/lexus/ -name "*.jpg" -exec cp "{}" ./cars/lexus \;
!find ../input/commoncars/Car\ Model/Toyota/ -name "*.jpg" -exec cp "{}" ./cars/toyota \;
!find ../input/commoncars/Car\ Model/Hyundai/ -name "*.jpg" -exec cp "{}" ./cars/hyundai \;
# for working with files 
import glob
import os
import shutil
import itertools  
from tqdm import tqdm

# for working with images
from PIL import Image
import numpy as np
import pandas as pd
from skimage import transform
import matplotlib.pyplot as plt
import cv2 as cv
import scipy.io
import random

# tensorflow stuff
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, BatchNormalization, GlobalAveragePooling2D, Add
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers, optimizers, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.activations import relu, softmax



# for evaluation
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

train_datagen=ImageDataGenerator(rotation_range=15,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 preprocessing_function=preprocess_input,validation_split=0.2)



train_generator=train_datagen.flow_from_directory(
    directory="cars/",
    batch_size=16,
    seed=42,
    target_size=(224,224),subset='training')


val_generator=train_datagen.flow_from_directory(
    directory="cars/",
    batch_size=16,
    seed=42,
    target_size=(224,224), subset='validation')


IMAGE_SIZE = 224
# Base model with MobileNetV2
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,alpha = .5,
                                               include_top=False, 
                                               weights='imagenet')

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dropout(.6)(x)
prediction_layer = tf.keras.layers.Dense(3, activation='softmax')(x)

learning_rate = 0.001

model=Model(inputs=base_model.input,outputs=prediction_layer)

for layer in model.layers[:80]:
    layer.trainable=False
for layer in model.layers[80:]:
    layer.trainable=True
# 

optimizer=tf.keras.optimizers.Adam(lr=learning_rate,clipnorm=0.0001)
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

lr_metric = get_lr_metric(optimizer)

model.compile(optimizer=optimizer,
             loss='categorical_crossentropy',
             metrics=['accuracy'])

#reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1, patience=1, verbose=1)

model.fit(train_generator,validation_data=val_generator,
          steps_per_epoch=80,validation_steps = 15,
          epochs=50,verbose=1)

val_generator.class_indices
scoreSeg = model.evaluate_generator(val_generator)
print("Accuracy = ",scoreSeg[1])


k = 0
for i,j in val_generator:
    print(i.shape, j.shape)
    p = model.predict(i)
    p = p.argmax(-1)
    t = j.argmax(-1)
    print(classification_report(t,p))
    print(confusion_matrix(t,p))
    k = k + 1
    if k > 3:
        break;
tf.keras.models.save_model(
    model,
    "threecars1_half87.h5"
)
model.predict(i)
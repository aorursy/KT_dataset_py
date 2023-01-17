! unzip -q "/kaggle/input/aerial-cactus-identification/train.zip"
! cp /kaggle/input/aerial-cactus-identification/train.csv .
# See Data 

import os

import cv2

from PIL import Image 

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("./train.csv")

df["has_cactus"] = df.has_cactus.astype("str")

images = df.sample(n=9)

for index, _image in enumerate(images.iterrows(), 1):

    image = _image[1]["id"]

    cactus = _image[1]["has_cactus"]

    image = cv2.imread("./train/" + image)

    plt.subplot(3,3,index)

    plt.figtext(0.99, 0.01, str(image.shape)) 

    plt.imshow(image)

    plt.title(cactus)

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

vgg_model = VGG16(include_top=False, input_shape=(32, 32, 3), classes=2)
img = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

train_gen = img.flow_from_dataframe(df,"/kaggle/working/train/", x_col="id", y_col="has_cactus", target_size=(32, 32), subset="training")

val_gen = img.flow_from_dataframe(df,"/kaggle/working/train/", x_col="id", y_col="has_cactus", target_size=(32, 32), subset="validation")
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras.optimizers import SGD, RMSprop

from tensorflow.keras.losses import categorical_crossentropy
for layer in vgg_model.layers:

    layer.trainable = False
vgg_output = vgg_model.output

flatten = Flatten()(vgg_output)

dense_1 = Dense(100, activation="relu")(flatten)

dense_2 = Dense(2, activation="softmax")(dense_1)
final_model = Model(inputs=vgg_model.input, outputs=dense_2)
final_model.summary()
for layer in final_model.layers:

    print(layer.name, layer.trainable)
final_model.compile(SGD(), categorical_crossentropy, metrics=["acc"])
final_model_history = final_model.fit_generator(train_gen,epochs=10, validation_data=val_gen)
plt.plot(final_model_history.history["loss"],label="Loss")

plt.plot(final_model_history.history["val_loss"], label="Val loss")

plt.legend()
plt.plot(final_model_history.history["acc"],label="Acc")

plt.plot(final_model_history.history["val_acc"], label="Val acc")

plt.legend()
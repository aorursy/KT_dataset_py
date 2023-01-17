# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import random
import urllib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
from PIL import Image

import tensorflow as tf
import tensorflow.keras as keras
from functools import partial, wraps, reduce
from tensorflow.keras.layers import (Conv2D, MaxPool2D, BatchNormalization, InputLayer, LeakyReLU, Dense, 
Flatten, Dropout, ReLU, SeparableConv2D, AveragePooling2D, GlobalAveragePooling2D)
# from tensorflow_addons.metrics import F1Score
from tensorflow.keras.regularizers import l2
import torch

torch.manual_seed(0)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
df = pd.read_json("/kaggle/input/vehicle-number-plate-detection/Indian_Number_plates.json", lines=True)
df.head()
os.mkdir("Indian Number Plates")
dataset = dict()
dataset["image_name"] = list()
dataset["image_width"] = list()
dataset["image_height"] = list()
dataset["top_x"] = list()
dataset["top_y"] = list()
dataset["bottom_x"] = list()
dataset["bottom_y"] = list()

counter = 0
for index, row in df.iterrows():
    img = urllib.request.urlopen(row["content"])
    img = Image.open(img)
    img = img.convert('RGB')
    img.save("Indian Number Plates/licensed_car{}.jpeg".format(counter), "JPEG")
    
    dataset["image_name"].append("licensed_car{}".format(counter))
    
    data = row["annotation"]
    
    dataset["image_width"].append(data[0]["imageWidth"])
    dataset["image_height"].append(data[0]["imageHeight"])
    dataset["top_x"].append(data[0]["points"][0]["x"])
    dataset["top_y"].append(data[0]["points"][0]["y"])
    dataset["bottom_x"].append(data[0]["points"][1]["x"])
    dataset["bottom_y"].append(data[0]["points"][1]["y"])
    
    counter += 1
print("Downloaded {} car images.".format(counter))
df = pd.DataFrame(dataset)
df.head()
df.to_csv("indian_license_plates.csv", index=False)
df = pd.read_csv("indian_license_plates.csv")
df["image_name"] = df["image_name"] + ".jpeg"
df.drop(["image_width", "image_height"], axis=1, inplace=True)
df.head()
print('Train data : ', df.shape[0])
lucky_test_samples = np.random.randint(0, len(df), 5)
reduced_df = df.drop(lucky_test_samples, axis=0)
lucky_test_samples
WIDTH = 224
HEIGHT = 224
CHANNEL = 3

def show_img(index):
    image = cv2.imread("Indian Number Plates/" + df["image_name"].iloc[index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(WIDTH, HEIGHT))

    tx = int(df["top_x"].iloc[index] * WIDTH)
    ty = int(df["top_y"].iloc[index] * HEIGHT)
    bx = int(df["bottom_x"].iloc[index] * WIDTH)
    by = int(df["bottom_y"].iloc[index] * HEIGHT)

    image = cv2.rectangle(image, (tx, ty), (bx, by), (0, 0, 255), 1)
    plt.imshow(image)
    plt.show()
show_img(150)
reduced_df
len(reduced_df)
def preprocess_img(path, box_cord):
    img = tf.io.read_file("Indian Number Plates/"+path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=(WIDTH, HEIGHT))
    return img, box_cord
TRAIN_DATA = 220

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((reduced_df['image_name'][:TRAIN_DATA],
                                              reduced_df.loc[:,'top_x':][:TRAIN_DATA].to_numpy()))

train_ds = (train_ds
            .shuffle(TRAIN_DATA)
            .map(preprocess_img, num_parallel_calls=AUTOTUNE)
            .batch(32)
            .prefetch(buffer_size=AUTOTUNE)
            )

val_ds = tf.data.Dataset.from_tensor_slices((reduced_df['image_name'][TRAIN_DATA:],
                                              reduced_df.loc[:,'top_x':][TRAIN_DATA:].to_numpy()))

val_ds = (val_ds
            .shuffle(len(reduced_df)-TRAIN_DATA)
            .map(preprocess_img, num_parallel_calls=AUTOTUNE)
            .batch(32)
            .prefetch(buffer_size=AUTOTUNE)
            )
next(iter(train_ds))
base_model = tf.keras.applications.ResNet50(input_shape=(HEIGHT, WIDTH, 3), 
                                            include_top=False, 
                                            weights='imagenet')
inputs = keras.Input(shape=(HEIGHT, WIDTH, 3))
resnet = base_model(inputs)
pool1 = GlobalAveragePooling2D()(resnet)
flatten = Flatten()(pool1)
# drop = Dropout(0.4)(flatten)
dense1 = Dense(1024, activation='relu')(flatten)
# drop = Dropout(0.4)(dense1)
outputs = Dense(4)(dense1)

model = keras.Model(inputs=inputs, outputs=outputs, name='MeNet_Model')

# set the resnet model to trainable
print(model.layers[1])
model.layers[1].trainable = False
model.summary()
adam_opt = tf.keras.optimizers.Adam()

model.compile(optimizer=adam_opt, 
              loss=tf.keras.losses.MeanSquaredError())
history = model.fit(train_ds,
                   validation_data=val_ds,
                   epochs=50)
for i in df.loc[lucky_test_samples].to_numpy():
    print(i[0])
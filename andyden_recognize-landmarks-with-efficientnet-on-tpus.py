import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import tensorflow.keras as K

import cv2

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

#!pip install efficientnet

import efficientnet.tfkeras as efn
train = pd.read_csv("../input/landmark-recognition-2020/train.csv")

train
num_classes = len(train["landmark_id"].unique()) #81313 unique landmarks

num_classes
train["path"] = train["id"].map(lambda x: "../input/landmark-recognition-2020/train/" + 

                                x[0] + "/" + x[1] + "/" + x[2] + "/" + x + ".jpg")

train
train["landmark_id"].value_counts().describe()
plt.style.use("ggplot")

sns.distplot(train["landmark_id"].value_counts())
sns.set_style('whitegrid', {'axes.grid' : False})

plt.figure(figsize=(20,10))



sample = train.sample(n=16).reset_index()



for i in range(16):

    plt.subplot(4, 4, i+1)

    img = cv2.imread(sample["path"][i])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(train["path"], train["landmark_id"], test_size=0.3)



# tf.dataset setting

AUTOTUNE = tf.data.experimental.AUTOTUNE



# training configuration

EPOCHS = 5

BATCH_SIZE = 32



# for model

IMAGE_SIZE = 128



def decode_image(filename, image_size=(IMAGE_SIZE, IMAGE_SIZE)):

    image = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    return image

    

def to_onehot(label):

    label = tf.one_hot(tf.cast(label, tf.int32), num_classes)

    label = tf.cast(label, tf.int32)

    return label
image_ds_train = tf.data.Dataset.from_tensor_slices(X_train).map(decode_image)

label_ds_train = tf.data.Dataset.from_tensor_slices(y_train).map(to_onehot)

image_ds_test = tf.data.Dataset.from_tensor_slices(X_test).map(decode_image)

label_ds_test = tf.data.Dataset.from_tensor_slices(y_test).map(to_onehot)



train_dataset = tf.data.Dataset.zip((image_ds_train, label_ds_train)).shuffle(1024).repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

valid_dataset = tf.data.Dataset.zip((image_ds_test, label_ds_test)).batch(BATCH_SIZE)
model = tf.keras.Sequential([

        efn.EfficientNetB3(

            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),

            weights='imagenet',

            include_top=False

        ),

        K.layers.GlobalAveragePooling2D(),

        K.layers.Dense(num_classes, activation='sigmoid')

    ])



model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['accuracy']

    )



model.summary()
STEPS_PER_EPOCH = y_train.shape[0] // BATCH_SIZE



history = model.fit(

    train_dataset, 

    batch_size=BATCH_SIZE,

    epochs=EPOCHS, 

    validation_data=valid_dataset,

    steps_per_epoch=STEPS_PER_EPOCH

#     callbacks=[],

)
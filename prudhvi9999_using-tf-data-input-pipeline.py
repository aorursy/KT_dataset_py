# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras import layers

from pathlib import Path

from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_df=pd.read_csv('/kaggle/input/hackerearth/dataset/train.csv')

test_df=pd.read_csv('/kaggle/input/hackerearth/dataset/test.csv')
img_col="Image"

class_col="Class"
train_img_path='/kaggle/input/hackerearth/dataset/Train Images/'

test_img_path='/kaggle/input/hackerearth/dataset/Test Images/'
EPOCHS = 10

BATCH_SIZE = 32

IMAGE_SIZE = 150

CHANNELS = 3

RANDOM_STATE = 1234

SEED = 5678

INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

NUM_CLASSES = train_df[class_col].nunique()
train_df.head()
label_to_index = dict((name, index) for index, name in enumerate(train_df[class_col].unique()))

label_to_index
# Reference

# https://www.tensorflow.org/tutorials/load_data/images

def preprocess_image(image, size=150, channels=3):

    image = tf.image.decode_jpeg(image, channels=channels)

    image = tf.image.resize(image, [size, size])

    image /= 255.0  # normalize to [0,1] range

    return image





def load_and_preprocess_image(path):

    image = tf.io.read_file(path)

    return preprocess_image(image)





def get_label_index(label_name):

    return label_to_index.get(label_name)





def get_image_path(part_path, ext=".jpg"):

    file_name = part_path

    return train_img_path+file_name



def get_img_id(img_path):

    return img_path.split("/")[-1].strip(".jpg")





def get_label(img_path):

    return train_df[train_df["Image"] == get_img_id(img_path)][class_col]
"""train_image_paths = list(map(str, train_img_path.glob("*.jpg")))

test_image_paths = list(map(str, test_img_path.glob("*.jpg")))"""
x_train, x_valid, y_train, y_valid = train_test_split(train_df[img_col],

                                                      train_df[class_col], 

                                                      stratify=train_df[class_col],

                                                      test_size=0.2,

                                                      random_state=RANDOM_STATE)
x_train.shape, x_valid.shape, y_train.shape, y_valid.shape
x_train.values[0]
get_image_path(x_train.values[0])
get_label_index(y_train.values[0])
def get_pair_ds(_images, _labels):

    path_ds = tf.data.Dataset.from_tensor_slices(list(map(get_image_path, _images.values)))

    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(list(map(get_label_index, _labels.values)),

                                                          tf.int64))

    return image_ds, label_ds





def apply_ds(_length, _image_ds, _label_ds):

    ds = tf.data.Dataset.zip((_image_ds, _label_ds))

    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=_length))

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds



train_ds = apply_ds(len(x_train), *get_pair_ds(x_train, y_train))

valid_ds = apply_ds(len(x_valid), *get_pair_ds(x_valid, y_valid))
train_ds
# Reference

# https://tfhub.dev/google/collections/efficientnet/1

feature_extractor_url = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"



def building_model(_input_shape, _num_classes):

    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,

                                             input_shape=_input_shape)

    feature_extractor_layer.trainable = False

    model = tf.keras.Sequential([

        feature_extractor_layer,

        layers.Dense(_num_classes, activation='softmax')

    ])

    return model
model = building_model(INPUT_SHAPE, NUM_CLASSES)

model.summary()
# callbacks

tensorboard = tf.keras.callbacks.TensorBoard("logs")

cp_callback = tf.keras.callbacks.ModelCheckpoint("logs/cp.cpkt", 

                                                 save_weights_only=True,

                                                 verbose=1)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",

                                                  patience=3)

_callbacks = [tensorboard, early_stopping, cp_callback]
tr_steps_per_epoch = tf.math.ceil(len(x_train) / BATCH_SIZE).numpy()

va_steps_per_epoch = tf.math.ceil(len(x_valid) / BATCH_SIZE).numpy()

tr_steps_per_epoch, va_steps_per_epoch
model.compile(

    optimizer=tf.keras.optimizers.Adam(),

    loss='sparse_categorical_crossentropy',

    metrics=['acc'])



history = model.fit(train_ds,

                    validation_data=valid_ds,

                    epochs=EPOCHS,

                    steps_per_epoch=tr_steps_per_epoch,

                    validation_steps=va_steps_per_epoch,

                    callbacks=_callbacks)
# loss, acc, val_loss, val_acc 

hist_keys = list(history.history.keys())
"""def gen():

    generator=img_gen.flow_from_dataframe(train_df,

                                               train_img_path,

                                               x_col='Image',

                                               y_col='Class',

                                               subset='training',

                                               class_mode='categorical',

                                               target_size=(150,150)

                                              )

    return generator"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_data_gen=ImageDataGenerator(rescale=1./255).flow_from_dataframe(test_df,

                                                                    test_img_path,

                                                                    x_col="Image",

                                                                    Y_col=None,

                                                                    target_size=(150,150),

                                                                    class_mode=None

                                                                   )
label_to_index={'Attire': 0, 'Decorationandsignage': 1, 'Food': 2, 'misc': 3}
def predictions(model):

    test_data_gen.reset()

    classes=model.predict(test_data_gen)

    predicted_class_indices=[np.argmax(i) for i in classes]

    labels = dict((v,k) for k,v in label_to_index.items())

    predictions = [labels[k] for k in predicted_class_indices]

    filenames=test_data_gen.filenames

    results=pd.DataFrame({"Image":filenames,

                      "Class":predictions})

    results.to_csv("tags.csv",index=False)
predictions(model)
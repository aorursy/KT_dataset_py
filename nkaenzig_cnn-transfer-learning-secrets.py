!wget https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz

!mkdir data

!tar zxf cifar10.tgz -C data/
# | Some global constants

LEARNING_RATE = 3e-4

BATCH_SIZE = 32

IMAGE_DIM = 128

NR_EPOCHS = 3



dataset_directory = 'data/cifar10'
from fastai import *

from fastai.vision import *
data = (ImageList.from_folder(dataset_directory)

                 .split_by_folder(train='train', valid='test')

                 .label_from_folder()

                 .transform(None, size=IMAGE_DIM)

                 .databunch(bs=64)

                 .normalize(imagenet_stats)

      )
learner = cnn_learner(data, models.resnet50, metrics=[accuracy], train_bn=False, wd=False, true_wd=False, bn_wd=False)

learner.fit(NR_EPOCHS, lr=LEARNING_RATE)
learner = cnn_learner(data, models.resnet50, metrics=[accuracy], train_bn=True, wd=False, true_wd=False, bn_wd=False)

learner.fit(NR_EPOCHS, lr=LEARNING_RATE)
import tensorflow as tf

from tensorflow import keras

AUTOTUNE = tf.data.experimental.AUTOTUNE

import pathlib

import os

from functools import partial

import random
data_root = pathlib.Path(dataset_directory)



# | Create dictionary for mapping classnames (strings) to integer

class_names = sorted([e.name for e in (data_root/'train').iterdir() if e.is_dir()])

class_names_to_label = {name: label for label, name in enumerate(class_names)}

label_to_class_names = {label: name for label, name in enumerate(class_names)}



# | Get all paths of train & test images

train_image_paths, train_image_labels, test_image_paths, test_image_labels = [], [], [], []

all_image_paths = list(data_root.glob('*/*/*.png'))

random.shuffle(all_image_paths)



for path in all_image_paths:

    if path.parent.parent.name == 'train':

        train_image_paths.append(str(path))

        train_image_labels.append(class_names_to_label[path.parent.name])

        

    elif path.parent.parent.name == 'test':

        test_image_paths.append(str(path))

        test_image_labels.append(class_names_to_label[path.parent.name])
def load_and_preprocess_image(image_path, label, img_shape):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_png(image, channels=3)

    image = tf.image.resize(image, img_shape)

    image = keras.applications.resnet_v2.preprocess_input(image)

    label = tf.cast(label, tf.float32)

    

    return image, label





def create_dataset(file_paths, labels, epochs=1, batch_size=64, buffer_size=10000, train=True):

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    dataset = dataset.map(partial(load_and_preprocess_image, img_shape=(IMAGE_DIM, IMAGE_DIM)), num_parallel_calls=AUTOTUNE)

    if train:

        #dataset = dataset.cache(filename='./cache.tf-data')

        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.repeat(epochs)

    dataset = dataset.batch(batch_size)

    if train:

        dataset = dataset.prefetch(AUTOTUNE)

    

    return dataset



def create_model(train_bn=False, dropout=0):

    base_model = keras.applications.resnet.ResNet50(weights="imagenet", include_top=False)

    

    if train_bn:

        for layer in base_model.layers:

            if layer.__class__.__name__ != "BatchNormalization":

                layer.trainable = False

    

    avg = keras.layers.GlobalAveragePooling2D()(base_model.output)

    mx = keras.layers.GlobalMaxPooling2D()(base_model.output)

    out = tf.keras.layers.Concatenate()([avg, mx])

    out = keras.layers.BatchNormalization()(out)

    out = keras.layers.Dropout(dropout)(out)

    out = keras.layers.Dense(512, activation="relu")(out)

    out = keras.layers.BatchNormalization()(out)

    out = keras.layers.Dropout(dropout)(out)

    out = keras.layers.Dense(10, activation="softmax")(out)

    

    model = keras.models.Model(inputs=base_model.input, outputs=out)

    

    optimizer = tf.keras.optimizers.Adam(lr=0.003)

    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    

    return model
steps_per_epoch = len(train_image_paths) // BATCH_SIZE

validation_steps = len(test_image_paths) // BATCH_SIZE



train_dataset = create_dataset(train_image_paths, train_image_labels, epochs=-1, batch_size=BATCH_SIZE, train=True)

validation_dataset = create_dataset(test_image_paths, test_image_labels, epochs=-1, batch_size=BATCH_SIZE, train=False)
model = create_model()



history = model.fit(train_dataset,

                    epochs = NR_EPOCHS,

                    steps_per_epoch = steps_per_epoch,

                    validation_data = validation_dataset, 

                    validation_steps = validation_steps,

                    callbacks = [])
model = create_model(train_bn=True)



history = model.fit(train_dataset,

                    epochs = NR_EPOCHS,

                    steps_per_epoch = steps_per_epoch,

                    validation_data = validation_dataset, 

                    validation_steps = validation_steps,

                    callbacks = [])
!rm -rf data
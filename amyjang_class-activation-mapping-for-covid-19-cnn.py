import re

import os

import random

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras

from kaggle_datasets import KaggleDatasets

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from sklearn.model_selection import train_test_split



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Device:', tpu.master())

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except:

    strategy = tf.distribute.get_strategy()

print('Number of replicas:', strategy.num_replicas_in_sync)

    

print(tf.__version__)
AUTOTUNE = tf.data.experimental.AUTOTUNE

GCS_PATH = KaggleDatasets().get_gcs_path("covid19-radiography-database")

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

IMAGE_SIZE = [180, 180]
filenames = tf.io.gfile.glob(str(GCS_PATH + '/COVID-19 Radiography Database/COVID-19/*'))

filenames.extend(tf.io.gfile.glob(str(GCS_PATH + '/COVID-19 Radiography Database/NORMAL/*')))

filenames.extend(tf.io.gfile.glob(str(GCS_PATH + '/COVID-19 Radiography Database/Viral Pneumonia/*')))



random.seed(1337)

tf.random.set_seed(1337)

random.shuffle(filenames)
train_filenames, test_filenames = train_test_split(filenames, test_size=0.1)

train_filenames, val_filenames = train_test_split(train_filenames, test_size=0.1)
COUNT_NORMAL = len([filename for filename in train_filenames if "NORMAL" in filename])

print("Normal images count in training set: " + str(COUNT_NORMAL))



COUNT_COVID = len([filename for filename in train_filenames if "/COVID-19/" in filename])

print("COVID-19 images count in training set: " + str(COUNT_COVID))



COUNT_PNEUMONIA = len([filename for filename in train_filenames if "Viral" in filename])

print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))
train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)

val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)

test_list_ds = tf.data.Dataset.from_tensor_slices(test_filenames)
TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()

print("Training images count: " + str(TRAIN_IMG_COUNT))



VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy()

print("Validating images count: " + str(VAL_IMG_COUNT))
CLASSES = ['NORMAL', 'COVID-19', 'Viral Pneumonia']
def get_label(file_path):

    # convert the path to a list of path components

    parts = tf.strings.split(file_path, os.path.sep)

    # The second to last is the class-directory

    return parts[-2] == CLASSES
def decode_img(img):

  # convert the compressed string to a 3D uint8 tensor

  img = tf.image.decode_png(img, channels=3)

  # Use `convert_image_dtype` to convert to floats in the [0,1] range.

  img = tf.image.convert_image_dtype(img, tf.float32)

  # resize the image to the desired size.

  return tf.image.resize(img, IMAGE_SIZE)
def process_path(file_path):

    label = get_label(file_path)

    # load the raw data from the file as a string

    img = tf.io.read_file(file_path)

    img = decode_img(img)

    return img, label
train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

test_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
def prepare_for_training(ds, cache=True):

    # This is a small dataset, only load it once, and keep it in memory.

    # use `.cache(filename)` to cache preprocessing work for datasets that don't

    # fit in memory.

    if cache:

        if isinstance(cache, str):

            ds = ds.cache(cache)

        else:

            ds = ds.cache()



    ds = ds.shuffle(buffer_size=1000)

    ds = ds.batch(BATCH_SIZE)



    if cache:

        ds = ds.prefetch(buffer_size=AUTOTUNE)



    return ds
train_ds = prepare_for_training(train_ds)

val_ds = prepare_for_training(val_ds)

test_ds = prepare_for_training(test_ds, False)
def show_batch(image_batch, label_batch):

    plt.figure(figsize=(10,10))

    for n in range(25):

        ax = plt.subplot(5,5,n+1)

        plt.imshow(image_batch[n])

        plt.title(CLASSES[np.argmax(label_batch[n])])

        plt.axis("off")
image_batch, label_batch = next(iter(train_ds))

show_batch(image_batch.numpy(), label_batch.numpy())
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,

                                                  restore_best_weights=True)
with strategy.scope():

    reconstructed_model = keras.models.load_model("../input/test-model/xray_model.h5")

    reconstructed_model.pop()

    reconstructed_model.add(keras.layers.Dense(3, activation='softmax'))

    

    METRICS = [

        'accuracy',

        keras.metrics.Precision(name="precision"),

        keras.metrics.Recall(name="recall")

    ]

    

    reconstructed_model.compile(

        optimizer="adam",

        loss="categorical_crossentropy",

        metrics=METRICS,

    )
history = reconstructed_model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=20,

    callbacks=[early_stopping_cb]

)
reconstructed_model.evaluate(test_ds, return_dict=True)
reconstructed_model.summary()
# last convolution block of the model

reconstructed_model.layers[7].layers
def get_img_array(img_path, size=IMAGE_SIZE):

    img = keras.preprocessing.image.load_img(img_path, target_size=size)

    # `array` is a float32 NumPy array

    array = keras.preprocessing.image.img_to_array(img)

    # We add a dimension to transform our array into a "batch"

    # of size (1, 180, 180, 3)

    array = np.expand_dims(array, axis=0) / 255.0

    return array
def make_gradcam_heatmap(img_array, model):

    # First, we create a model that maps the input image to the activations

    # of the last conv layer

    last_conv_layer = model.layers[7]

    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    

    # Mark the classifying layers

    classifier_layers = model.layers[-5:]



    # Second, we create a model that maps the activations of the last conv

    # layer to the final class predictions

    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])

    x = classifier_input

    for classifier_layer in classifier_layers:

        x = classifier_layer(x)

    classifier_model = keras.Model(classifier_input, x)



    # Then, we compute the gradient of the top predicted class for our input image

    # with respect to the activations of the last conv layer

    with tf.GradientTape() as tape:

        # Compute activations of the last conv layer and make the tape watch it

        last_conv_layer_output = last_conv_layer_model(img_array)

        tape.watch(last_conv_layer_output)

        # Compute class predictions

        preds = classifier_model(last_conv_layer_output)

        top_pred_index = tf.argmax(preds[0])

        top_class_channel = preds[:, top_pred_index]



    # This is the gradient of the top predicted class with regard to

    # the output feature map of the last conv layer

    grads = tape.gradient(top_class_channel, last_conv_layer_output)



    # This is a vector where each entry is the mean intensity of the gradient

    # over a specific feature map channel

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))



    # We multiply each channel in the feature map array

    # by "how important this channel is" with regard to the top predicted class

    last_conv_layer_output = last_conv_layer_output.numpy()[0]

    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):

        last_conv_layer_output[:, :, i] *= pooled_grads[i]



    # The channel-wise mean of the resulting feature map

    # is our heatmap of class activation

    heatmap = np.mean(last_conv_layer_output, axis=-1)



    # For visualization purpose, we will also normalize the heatmap between 0 & 1

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap
def superimposed_cam(file_path):

    # Prepare image

    img_array = get_img_array(file_path)



    # Generate class activation heatmap

    heatmap = make_gradcam_heatmap(

        img_array, reconstructed_model

    )



    # Rescale the original image

    img = img_array * 255



    # We rescale heatmap to a range 0-255

    heatmap = np.uint8(255 * heatmap)



    # We use jet colormap to colorize heatmap

    jet = cm.get_cmap("jet")



    # We use RGB values of the colormap

    jet_colors = jet(np.arange(256))[:, :3]

    jet_heatmap = jet_colors[heatmap]



    # We create an image with RGB colorized heatmap

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)

    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))

    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)



    # Superimpose the heatmap on original image

    superimposed_img = jet_heatmap * 0.4 + img

    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img[0])

    

    return superimposed_img, CLASSES[np.argmax(reconstructed_model.predict(img_array))]
covid_filenames = tf.io.gfile.glob('../input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19/*')

pneumonia_filenames = tf.io.gfile.glob('../input/covid19-radiography-database/COVID-19 Radiography Database/Viral Pneumonia/*')
plt.figure(figsize=(20,20))

for n in range(10):

    ax = plt.subplot(5,5,n+1)

    img, pred = superimposed_cam(covid_filenames[n])

    plt.imshow(img)

    plt.title(pred)

    plt.axis("off")

for n in range(15, 25):

    ax = plt.subplot(5,5,n+1)

    img, pred = superimposed_cam(pneumonia_filenames[n])

    plt.imshow(img)

    plt.title(pred)

    plt.axis("off")
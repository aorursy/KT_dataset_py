from kaggle_datasets import KaggleDatasets

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
GCS_DS_PATH = KaggleDatasets().get_gcs_path("colorization-data")
GCS_DS_PATH
import tensorflow as tf
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except ValueError: 

    strategy = tf.distribute.experimental.MirroredStrategy()

    

print("Number of accelerators: ", strategy.num_replicas_in_sync)
EPOCHS = 50

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH+'/ColorizationData/Train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH+'/ColorizationData/Val/*.tfrec')
import numpy as np

import cv2

import matplotlib.pyplot as plt

import os
AUTO = tf.data.experimental.AUTOTUNE
def read_tfrecord(example):

    tfrec_format = {

        "X" : tf.io.FixedLenFeature([224,224,3], tf.float32),

        "Y" : tf.io.FixedLenFeature([50176,2], tf.float32)

    }

    example = tf.io.parse_single_example(example, tfrec_format)

    x = example['X']

    y = tf.one_hot(tf.cast(example['Y'],tf.int64), depth=32)

    y = {"output1":y[:,0], "output2":y[:,1]}

    

    return x,y
def load_dataset(filenames):

    tf_op = tf.data.Options()

    tf_op.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.with_options(tf_op)

    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

    return dataset
def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES)

    dataset = dataset.repeat() 

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) 

    return dataset
def get_validation_dataset():

    dataset = load_dataset(VALIDATION_FILENAMES)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) 

    return dataset
import re
def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} validation images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES))
# print("Training data shapes:")

# for x, y in get_training_dataset().take(3):

#     print(x.numpy().shape, y["output1"].numpy().shape, y["output2"].numpy().shape)

    

# print("Validation data shapes:")

# for x, y in get_validation_dataset().take(3):

#     print(x.numpy().shape, y["output1"].numpy().shape, y["output2"].numpy().shape)
# from keras.applications.vgg16 import VGG16

# from keras.layers import Add, Conv2D, UpSampling2D, BatchNormalization, Input, Concatenate, Reshape, Activation, Conv2DTranspose
from keras.models import Model

# with strategy.scope():

#     vgg = VGG16()

#     for i in range(len(vgg.layers)):

#         vgg.layers[i].trainable = False



#     #conv1 = Conv2D(256,(1,1),activation="relu",padding="same")(vgg.layers[14].input)

#     #upscale1 = UpSampling2D()(conv1)

#     batchnorm1 = BatchNormalization()(vgg.layers[14].input)

#     upscale1 = Conv2DTranspose(256, (1,1), activation="relu",strides=(2,2), padding="same")(batchnorm1)



#     batchnorm2 = BatchNormalization()(vgg.layers[10].input)

#     add1 = Add(name="Add1")([upscale1,batchnorm2])

#     #conv2 = Conv2D(128,(3,3),activation="relu",padding="same")(add1)

#     #upscale2 = UpSampling2D()(conv2)

#     upscale2 = Conv2DTranspose(128, (3,3), activation="relu", strides=(2,2), padding="same")(add1)





#     batchnorm3 = BatchNormalization()(vgg.layers[6].input)

#     add2 = Add(name="Add2")([upscale2,batchnorm3])

#     #conv3 = Conv2D(64,(3,3),activation="relu",padding="same")(add2)

#     #upscale3 = UpSampling2D()(conv3)

#     upscale3 = Conv2DTranspose(64, (3,3), activation="relu", strides=(2,2), padding="same")(add2)



#     #upscale4 = UpSampling2D(size=(4,4))(upscale1)

#     #upscale5 = UpSampling2D()(upscale2)

#     upscale4 = Conv2DTranspose(64,(1,1),activation="relu",strides=(4,4),padding="same")(upscale1)

#     upscale5 = Conv2DTranspose(64,(1,1), activation="relu",strides=(2,2), padding="same")(upscale2)



#     batchnorm4 = BatchNormalization()(vgg.layers[3].input)

#     concat = Add()([upscale3, upscale4, upscale5, batchnorm4])

#     conv4 = Conv2D(256,(3,3),activation="relu",padding="same")(concat)

#     conv5 = Conv2D(64,(3,3),activation="relu",padding="same")(conv4)

#     conv6 = Conv2D(64,(3,3),activation="relu",padding="same")(conv5)



#     u_conv = Conv2D(32, (3,3), padding="same")(conv6)

#     u_flatten = Reshape((50176,32))(u_conv)

#     u_act = Activation("softmax", name="output1")(u_flatten)



#     v_conv = Conv2D(32, (3,3), padding="same")(conv6)

#     v_flatten = Reshape((50176,32))(v_conv)

#     v_act = Activation("softmax",name="output2")(v_flatten)

    

#     color_model = Model(inputs=vgg.layers[0].output,outputs=[u_act, v_act])

    

#     color_model.compile(optimizer="adam",loss={"output1":"categorical_crossentropy", "output2":"categorical_crossentropy"},metrics=["accuracy"])
# color_model.summary()
# color_model._layers = [layer for layer in color_model._layers if not isinstance(layer, dict)]
from keras.models import load_model
with strategy.scope():

    color_model = load_model("../input/trained-model/color_modelClass.h5")
# color_model._layers = [layer for layer in color_model._layers if not isinstance(layer, dict)]
# from tensorflow.keras.utils import plot_model
# plot_model(color_model, show_shapes=True)
history = color_model.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=get_validation_dataset())
color_model.save("color_modelClass.h5")
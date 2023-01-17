!pip install efficientnet
import math, re, os



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

import tensorflow as tf

import tensorflow.keras.layers as L





from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import AveragePooling2D

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import ResNet50, InceptionResNetV2, DenseNet201, ResNet50V2

from sklearn.metrics import classification_report

from tensorflow.keras.callbacks import ModelCheckpoint



# import efficientnet.tfkeras as efn

from tensorflow.keras.applications import ResNet50

from sklearn import metrics

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint



import efficientnet.tfkeras as efn
AUTO = tf.data.experimental.AUTOTUNE

# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()
INPUT = "/kaggle/input/journey-springfield"

TRAIN_DIR = "/kaggle/input/journey-springfield/train/simpsons_dataset"

TEST_DIR = "/kaggle/input/journey-springfield/testset/testset"



EPOCHS = 25

BATCH_SIZE = 8 * strategy.num_replicas_in_sync

IM_Z = 784
df = pd.read_csv("/kaggle/input/journey-springfield/sample_submission.csv")

test_files = df["Id"].values
test_classes = {k:v for v, k in zip(sorted(os.listdir(TRAIN_DIR)), range(42))}

train_classes = {v:k for v, k in zip(sorted(os.listdir(TRAIN_DIR)), range(42))}



dirs = sorted(os.listdir(TRAIN_DIR))

train_paths = np.array(([GCS_DS_PATH + "/train/simpsons_dataset/" + i + "/" +  j for i in dirs for j in os.listdir(TRAIN_DIR + "/" + i)]))

tmp_labels = np.array([train_classes[j.split("/")[-2]] for j in train_paths])



train_labels = np.zeros((train_paths.shape[0], 42))

for i in range(train_paths.shape[0]):

    train_labels[i, tmp_labels[i]] = 1

print(train_paths.shape)



test_paths = np.array(([GCS_DS_PATH + "/testset/testset/" +  i for i in test_files]))



# train_paths, valid_paths, train_labels, valid_labels = train_test_split(

#     train_paths, train_labels, test_size=0.1, random_state=2020)
train_labels = train_labels.astype(int)
def decode_image(filename, label=None, image_size=(IM_Z, IM_Z)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_png(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    

#     print(label)

    if label is None:

        return image

    else:

        return image, label



def data_augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

#     image = tf.image.adjust_brightness(image, delta=0.2)

#     image = tf.image.adjust_contrast(image,2)

    

    if label is None:

        return image

    else:

        return image, label
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .cache()

    .map(data_augment, num_parallel_calls=AUTO)

    .repeat()

    .shuffle(512)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



# valid_dataset = (

#     tf.data.Dataset

#     .from_tensor_slices((valid_paths, valid_labels))

#     .map(decode_image, num_parallel_calls=AUTO)

#     .batch(BATCH_SIZE)

#     .cache()

#     .prefetch(AUTO)

# )



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(test_paths)

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

)
LR_START = 0.00001

LR_MAX = 0.0001 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 15

LR_SUSTAIN_EPOCHS = 3

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



# rng = [i for i in range(EPOCHS)]

# y = [lrfn(x) for x in rng]

# plt.plot(rng, y)

# print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
with strategy.scope():

    model_efnB7 = tf.keras.Sequential([

            efn.EfficientNetB7(weights="imagenet", include_top=False, input_shape=(IM_Z, IM_Z, 3)),



            L.GlobalAveragePooling2D(),

            L.Flatten(name="flatten"),

            L.Dense(256, activation="relu"),

            L.Dropout(0.5),

            L.Dense(42, activation='softmax')

    ])

    

    opt = Adam(lr=1e-4, decay=1e-4 / EPOCHS)



    model_efnB7.compile(

        optimizer=opt,

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

#     model.summary()
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE 



history = model_efnB7.fit(

    train_dataset, 

    epochs=EPOCHS, 

    steps_per_epoch=STEPS_PER_EPOCH,

    callbacks=[lr_callback],

    

#     validation_data=valid_dataset

)
# with strategy.scope():

#     model_efnB6 = tf.keras.Sequential([

#             efn.EfficientNetB6(weights="imagenet", include_top=False, input_shape=(IM_Z, IM_Z, 3)),



#             L.GlobalAveragePooling2D(),

#             L.Flatten(name="flatten"),

#             L.Dense(256, activation="relu"),

#             L.Dropout(0.5),

#             L.Dense(42, activation='softmax')

#     ])

    

#     opt = Adam(lr=1e-4, decay=1e-4 / EPOCHS)



#     model_efnB6.compile(

#         optimizer=opt,

#         loss = 'categorical_crossentropy',

#         metrics=['categorical_accuracy']

#     )

# #     model.summary()
# STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE 



# history = model_efnB6.fit(

#     train_dataset, 

#     epochs=EPOCHS, 

#     steps_per_epoch=STEPS_PER_EPOCH,

#     callbacks=[lr_callback],

# #     validation_data=valid_dataset

# )
# prob = (model_efnB7.predict(test_dataset, verbose=1) + model_efnB6.predict(test_dataset, verbose=1))/2

prob = model_efnB7.predict(test_dataset, verbose=1)
df.Expected = np.array([test_classes[i] for i in np.argmax(prob, axis=1)])

df
df.to_csv("submission.csv", index=False)
!pip install -q tensorflow==2.3.0 # Use 2.3.0 for built-in EfficientNet

!pip install -q git+https://github.com/keras-team/keras-tuner@master # Use github head for newly added TPU support

!pip install -q cloud-tpu-client # Needed for sync TPU version



!pip install -U tensorflow-gcs-config==2.3.0 # Needed for using private dataset
import random, re, math

import numpy as np, pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf, tensorflow.keras.backend as K

from kaggle_datasets import KaggleDatasets



print('Tensorflow version ' + tf.__version__)

import kerastuner as kt
# Detect hardware, return appropriate distribution strategy

try:

    # Sync TPU version

    from cloud_tpu_client import Client

    c = Client()

    c.configure_tpu_version(tf.__version__, restart_type='ifNeeded')

    

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None

    



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
# Configuration

IMAGE_SIZE = [256, 256]

EPOCHS_SEARCH = 5

EPOCHS_FINAL = 5

# SEED = 123

BATCH_SIZE = 32 * strategy.num_replicas_in_sync
from tensorflow.data.experimental import AUTOTUNE

base_path = KaggleDatasets().get_gcs_path('gld-v2-256')
import os

import functools





def create_dataset(file_pattern, allowed_labels, augmentation: bool = False, num_classes=None):

    # Select only dataset within a list of allowed labels

    if not num_classes:

        raise ValueError('num_classses must be set.')



    ignore_order = tf.data.Options()

    ignore_order.experimental_deterministic = False

    filenames = tf.io.gfile.glob(file_pattern)

    filenames = filenames



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE).shuffle(1000)



    # Create a description of the features.

    feature_description = {

        'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),

        'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),

        'image/channels': tf.io.FixedLenFeature([], tf.int64, default_value=0),

        'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),

        'image/id': tf.io.FixedLenFeature([], tf.string, default_value=''),

        'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),

        'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),

        'image/class/label': tf.io.FixedLenFeature([], tf.int64, default_value=0),

    }



    parse_func = functools.partial(

        _parse_example,

        name_to_features=feature_description,

        augmentation=augmentation

    )

    dataset = dataset.with_options(ignore_order)

    dataset = dataset.map(parse_func, num_parallel_calls=AUTOTUNE)



    def label_predicate(x, y):

        return tf.greater(tf.reduce_sum(tf.cast(tf.equal(allowed_labels, y), tf.float32)), 0.)



    def relabel(x, y):

        y = tf.reduce_min(tf.where(tf.equal(allowed_labels, y)))

        return x, tf.one_hot(y, num_classes)



    dataset = dataset.filter(label_predicate)

    dataset = dataset.map(relabel, num_parallel_calls=AUTOTUNE)

    dataset = dataset.repeat()



    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset



def _parse_example(example, name_to_features, augmentation):

    parsed_example = tf.io.parse_single_example(example, name_to_features)



    image = parsed_example['image/encoded']

    image = tf.io.decode_jpeg(image)

    image = tf.cast(image, tf.float32)

    image = tf.image.resize(image, IMAGE_SIZE)

    image.set_shape([*IMAGE_SIZE, 3])



    label = tf.cast(parsed_example['image/class/label'], tf.int64)

    return image, label
# original labelling

training_csv_path = os.path.join(base_path, "train.csv")

train_csv = pd.read_csv(str(training_csv_path))



# original labelling

clean_training_csv_path = os.path.join(base_path, "train_clean.csv")

clean_train_csv = pd.read_csv(str(clean_training_csv_path))

###

orig_unique_landmark_ids = clean_train_csv["landmark_id"].tolist()

print('max label:', max(orig_unique_landmark_ids))

###
landmark_ids_occurance = [len(x.split(" ")) for x in clean_train_csv["images"]]

# The labelling used in tfrecord is compressed, corresponding to 0 based id of clean_csv

compressed_landmark_ids_to_occurance = list(enumerate(landmark_ids_occurance))



#unique_landmark_ids = [x[0] for x in unique_landmark_ids_to_occurance]



allowed_labels = [x[0] for x in compressed_landmark_ids_to_occurance if x[1] >= 25]

allowed_labels = tf.convert_to_tensor(allowed_labels, dtype=tf.int64)



num_samples = sum([x for x in landmark_ids_occurance if x >= 25])

NUM_CLASSES = len([x for x in landmark_ids_occurance if x >= 25])





# unique_landmark_ids_occurance = tf.convert_to_tensor(unique_landmark_ids_occurance)

print(num_samples)

steps_per_epoch = int(num_samples / BATCH_SIZE)

_num_samples = steps_per_epoch * BATCH_SIZE
# train_tf_records = os.path.join(base_path, 'train*128')

# val_tf_records = os.path.join(base_path, 'val*128')

all_tf_records = os.path.join(base_path, '*128')



# ds_train = create_dataset(train_tf_records,

#                           allowed_labels,

#                           num_classes = NUM_CLASSES)



# ds_val = create_dataset(val_tf_records,

#                         allowed_labels,

#                         num_classes = NUM_CLASSES)



ds_all = create_dataset(all_tf_records,

                        allowed_labels,

                        num_classes = NUM_CLASSES)
# for img, lbl in ds_train.shuffle(10).take(1):

#     plt.imshow(tf.cast(img[0], tf.int32))
# from kerastuner.applications.efficientnet import HyperEfficientNet

# class MyHyperEfficientNet(HyperEfficientNet):

#     def _compile(self, model, hp):

        

#         for l in model.layers:

#             # For efficientnet implementation we use, layers in the

#             # Feature extraction part of model all have 'block' in name.

#             if 'block' in l.name:

#                 l.trainable = False

                

#         super(MyHyperEfficientNet, self)._compile(model, hp)
# # Define HyperModel using built-in application

# from kerastuner.applications.efficientnet import HyperEfficientNet

# hm = HyperEfficientNet(input_shape=[*IMAGE_SIZE, 3] , classes=NUM_CLASSES)



# # Optional: Restrict default hyperparameters.

# # To take effect, pass this `hp` instance when constructing tuner as `hyperparameters=hp`

# from kerastuner.engine.hyperparameters import HyperParameters

# hp = HyperParameters()

# hp.Choice('version', ['B0', 'B1', 'B2', 'B3']) #restrict choice of EfficientNet version from B0-B7 to B0-B4
# Define Oracle

# oracle = kt.tuners.randomsearch.RandomSearchOracle(

#     objective='val_accuracy',

#     max_trials=5,

#     hyperparameters=hp,

# )



# # Initiate Tuner

# tuner = kt.engine.tuner.Tuner(

#     hypermodel=hm,

#     oracle=oracle,

#     distribution_strategy=strategy, # This strategy's scope is used for building each model during the search.

#     directory='landmark',

#     project_name='randomsearch_efficientnet',

# )

# tuner.search_space_summary()
val_split = 0.2

num_val_samples = int(num_samples * val_split)

num_train_samples = int(num_samples * (1 - val_split))



num_train_batches = num_train_samples // BATCH_SIZE

num_val_batches = num_val_samples // BATCH_SIZE
# tuner.search(ds_train,

#              epochs=EPOCHS_SEARCH,

#              validation_data=ds_val,

#              steps_per_epoch=num_train_batches,

#              validation_steps=num_val_batches,

#              verbose=1)
# tuner.results_summary()

# model = tuner.get_best_models()[0]
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

from tensorflow.keras.applications import EfficientNetB7



with strategy.scope():

    input = tf.keras.layers.Input(shape = (*IMAGE_SIZE,3))

    

    # Create and Compile Model and show Summary

    effnet_model = EfficientNetB7(weights = "imagenet", include_top = False, input_tensor = input, pooling = 'avg', classes = None)

    

    X = tf.keras.layers.Dropout(0.25)(effnet_model.output)

    X = tf.keras.layers.Dense(1024, activation = 'relu')(X)

    X = tf.keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Dropout(0.25)(X)

    preds = tf.keras.layers.Dense(NUM_CLASSES, activation = 'softmax')(X)

    

    # Create Final Model

    model = tf.keras.Model(inputs = effnet_model.input, outputs = preds)



    # UnFreeze all layers

    for layer in model.layers:

        layer.trainable = True

        

    opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4 / EPOCHS_FINAL)



    model.compile(

        optimizer=opt,

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )
# from tensorflow.keras.applications import EfficientNetB2





# with strategy.scope():

#     model = tf.keras.Sequential([

#             EfficientNetB2(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3)),



#             tf.keras.layers.GlobalAveragePooling2D(),

#             tf.keras.layers.Flatten(name="flatten"),

#             tf.keras.layers.Dense(256, activation="relu"),

#             tf.keras.layers.Dropout(0.5),

#             tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')

#     ])

    

#     opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4 / EPOCHS_FINAL)



#     model.compile(

#         optimizer=opt,

#         loss = 'categorical_crossentropy',

#         metrics=['categorical_accuracy']

#     )

# #     model.summary()
# K.clear_session()

# model = tf.keras.models.load_model('../input/landmark-tpu/model.h5')
# Train the best model with all data

model.fit(ds_all,

          epochs=EPOCHS_FINAL,

#           batch_size=BATCH_SIZE,

          steps_per_epoch=num_train_batches + num_val_batches,

#           callbacks=[tf.keras.callbacks.ReduceLROnPlateau(),lr_callback],

          callbacks=[lr_callback],

          verbose=2

               )
os.chdir("/kaggle/working/")

model.save("model.h5")
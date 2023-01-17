!pip install --quiet tensorflow-datasets
import os, time

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

import tensorflow_datasets as tfds

from kaggle_datasets import KaggleDatasets

import visiontools
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
if tpu:

    DATA_DIR = KaggleDatasets().get_gcs_path('cars196')

else:

    DATA_DIR = '/kaggle/input/cars196'



ds, ds_info = tfds.load('cars196',

                        split='train',

                        with_info=True,

                        download=False,

                        data_dir=DATA_DIR)



fig = tfds.show_examples(ds_info=ds_info, ds=ds)
from visiontools import make_preprocessor, make_augmentor



preprocessor = make_preprocessor(size=[192, 192])



augmentor = make_augmentor(hue_delta=0.25,

                           saturation_range=[0.1, 3.0],

                           horizontal_flip=True)



rows = 4; cols = 4

ds = tfds.load('cars196', split='train',

               download=False, data_dir=DATA_DIR,

               as_supervised=True)

examples = list(tfds.as_numpy(ds.take(rows * cols)))



plt.figure(figsize=(15, (15 * rows) // cols))

for i, (image, label) in enumerate(examples):

    image, _ = preprocessor(image, label)

    image, _ = augmentor(image, label)

    plt.subplot(rows, cols, i+1)

    plt.axis('off')

    plt.imshow(image)

plt.show()
(ds_train, ds_validation), ds_info = tfds.load('cars196',

                                               download=False,

                                               data_dir=DATA_DIR,

                                               split=['train', 'test'],

                                               as_supervised=True,

                                               shuffle_files=True,

                                               with_info=True)
NUM_LABELS = 196

NUM_TRAINING_IMAGES = ds_info.splits['train'].num_examples

SHUFFLE_BUFFER =  NUM_TRAINING_IMAGES // 4

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

SIZE = [512, 512]

AUTO = tf.data.experimental.AUTOTUNE



preprocess = make_preprocessor(size=SIZE)

augment = make_augmentor(horizontal_flip=True,

                        hue_delta=0.2,

                        saturation_range=[0, 2],

                        contrast_range=[0.5, 2])



# Training Pipeline

ds_train = (

    ds_train.map(preprocess, AUTO) # preprocess before caching

    .cache() # cache before augmenting

    .repeat() # why is this needed?

    .shuffle(SHUFFLE_BUFFER) # shuffle before batching

    .batch(BATCH_SIZE) # batch before augmenting

    .map(augment, AUTO)

    .prefetch(AUTO) # prefetch last

)



# Validation Pipeline

# since the training set is shuffled, we can cache the batches for the

# validation set

ds_validation = (

    ds_validation.map(preprocess, AUTO)

    .batch(BATCH_SIZE)

    .cache() 

    .prefetch(AUTO)

)



# TODO - Tune and profile these. Customize jpeg decoding and bbox

# crop/resize.
with strategy.scope():

    pretrained_model = tf.keras.applications.Xception(weights='imagenet',

                                                      include_top=False,

                                                      input_shape=[*SIZE, 3])

    pretrained_model.trainable = True

    model = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(NUM_LABELS,

                              activation='softmax',

                              dtype=tf.float32),

    ])



model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy'],

)



model.summary()
from visiontools import exponential_lr



# Fine tuning is sensitive to learning rate

lr_callback = (

    tf.keras

    .callbacks

    .LearningRateScheduler(lambda epoch: exponential_lr(epoch),

                           verbose=True)

)



es_callback = (

    tf.keras

    .callbacks

    .EarlyStopping(monitor='val_loss',

                   min_delta=1e-4,

                   patience=3,

                   restore_best_weights=True)

)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

EPOCHS = 80



start_time = time.time()

history = model.fit(ds_train,

                    validation_data=ds_validation,

                    epochs=EPOCHS,

                    steps_per_epoch=STEPS_PER_EPOCH,

                    callbacks=[lr_callback,

                               es_callback])

final_accuracy = history.history["val_sparse_categorical_accuracy"][-5:]

print("FINAL ACCURACY MEAN-5: ", np.mean(final_accuracy))

print ("TRAINING TIME: ", time.time() - start_time, " sec")

model.save('/kaggle/working/densenet.h5')
# TODO - training visualizations
# TODO - predictions
# TODO - confusion matrix
# TODO - examine misclassified images
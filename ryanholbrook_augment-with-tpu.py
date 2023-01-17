!pip install -qU tensorflow-datasets

!pip install -qU tf-models-official
import tensorflow as tf

import tensorflow_datasets as tfds

from kaggle_datasets import KaggleDatasets

from tensorflow.raw_ops import ImageProjectiveTransformV2

from official.vision.image_classification import augment as transform
import tensorflow as tf



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

    DATA_DIR = KaggleDatasets().get_gcs_path('tensorflow-flowers')

else:

    DATA_DIR = '/kaggle/input/tensorflow-flowers'



(ds_train, ds_valid), ds_info = tfds.load(

    'tf_flowers',

    split=['train[:70%]', 'train[70%:]'],

    as_supervised=True,

    with_info=True,

    shuffle_files=True,

    data_dir=DATA_DIR,

)
import matplotlib.pyplot as plt



ROWS = 3

COLS = 4



plt.figure(figsize=(12, 12))

for i, (image, label) in enumerate(ds_train.take(ROWS*COLS)):

    angle = tf.random.uniform([], 0, 360, dtype=tf.float32)

    image = transform.rotate(image, angle)

    name = ds_info.features['label'].int2str(label)

    plt.subplot(ROWS, COLS, i+1)

    plt.title("{} ({})".format(name, label))

    plt.axis('off')

    plt.imshow(image)
auto_augment = transform.AutoAugment(translate_const=1)



ROWS = 3

COLS = 4



plt.figure(figsize=(12, 12))

for i, (image, label) in enumerate(ds_train.take(ROWS*COLS)):

    image = auto_augment.distort(image)

    name = ds_info.features['label'].int2str(label)

    plt.subplot(ROWS, COLS, i+1)

    plt.title("{} ({})".format(name, label))

    plt.axis('off')

    plt.imshow(image)

plt.tight_layout()

plt.show();
SIZE = [512, 512]



def preprocess(image, label):

    image = tf.image.resize(image, size=SIZE)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    return image, label



def augment(image, label):

    image = auto_augment.distort(image)

    return image, label
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

AUTO = tf.data.experimental.AUTOTUNE

NUM_TRAIN = int(ds_info.splits['train'].num_examples * 0.7)

SHUFFLE_BUFFER = NUM_TRAIN



ds_train = (ds_train

            .map(preprocess, AUTO)

            .cache()

            .shuffle(SHUFFLE_BUFFER)

            .repeat()

            .map(augment, AUTO)

            .batch(BATCH_SIZE, drop_remainder=True)

            .prefetch(AUTO))



ds_valid = (ds_valid

            .map(preprocess, AUTO)

            .cache()

            .batch(BATCH_SIZE)

            .prefetch(AUTO))
import tensorflow.keras as keras

import tensorflow.keras.layers as layers



NUM_CLASSES = ds_info.features['label'].num_classes



with strategy.scope():

    model = tf.keras.Sequential([

        layers.BatchNormalization(),

        layers.Conv2D(64, kernel_size=5, activation='relu'),

        layers.MaxPool2D(),

        

        layers.BatchNormalization(),

        layers.Conv2D(128, kernel_size=3, activation='relu'),

        layers.MaxPool2D(),

        

        layers.BatchNormalization(),

        layers.Conv2D(256, kernel_size=3, activation='elu'),

        layers.Conv2D(256, kernel_size=3, activation='elu'),

        layers.MaxPool2D(),

        

        layers.BatchNormalization(),

        layers.Conv2D(512, kernel_size=3, activation='elu'),

        layers.Conv2D(512, kernel_size=3, activation='elu'),

        layers.Conv2D(512, kernel_size=3, activation='elu'),

        layers.MaxPool2D(),

        

        layers.BatchNormalization(),

        layers.Conv2D(1024, kernel_size=3, activation='elu'),

        layers.Conv2D(1024, kernel_size=3, activation='elu'),

        layers.Conv2D(1024, kernel_size=3, activation='elu'),

        layers.MaxPool2D(),

        

        layers.Flatten(),

        layers.Dense(2048, activation='elu'),

#         layers.Dropout(0.5),

        layers.Dense(1024, activation='elu'),

#         layers.Dropout(0.5),

        layers.Dense(NUM_CLASSES, activation='softmax')

    ])

    model.compile(

        optimizer='adam',

        loss='sparse_categorical_crossentropy',

        metrics=['sparse_categorical_accuracy'],

    )
EPOCHS = 100

STEPS_PER_EPOCH = NUM_TRAIN // BATCH_SIZE



early_stopping = tf.keras.callbacks.EarlyStopping(patience=15, min_delta=0.001, restore_best_weights=True)



history = model.fit(

    ds_train,

    validation_data=ds_valid,

    epochs=EPOCHS,

    steps_per_epoch=STEPS_PER_EPOCH,

    callbacks=[early_stopping],

)
import pandas as pd



history_frame = pd.DataFrame(history.history)

history_frame.loc[:, ['loss', 'val_loss']].plot()

history_frame.loc[:, ['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].plot();
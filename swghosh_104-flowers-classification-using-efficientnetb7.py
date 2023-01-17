!pip uninstall -y Keras-Applications

!pip install git+https://github.com/keras-team/keras-applications.git
import tensorflow as tf

import kaggle_datasets as kd
import keras_applications as kapps



kapps._KERAS_BACKEND = tf.keras.backend

kapps._KERAS_LAYERS = tf.keras.layers

kapps._KERAS_MODELS = tf.keras.models

kapps._KERAS_UTILS = tf.keras.utils
IMAGE_SIZE = 512

GCS_PATH = kd.KaggleDatasets().get_gcs_path()

DS_PATH = '{gcs}/tfrecords-jpeg-{size}x{size}'.format(gcs=GCS_PATH, 

                                                      size=IMAGE_SIZE)



TRAIN_PATH, VAL_PATH, TEST_PATH = ['{tfrecs}/{split}'.format(tfrecs=DS_PATH, 

                                                            split=split) 

                                    for split in ('train', 'val', 'test')]

NUM_CLASSES = 104

BATCH_SIZE = 128



NUM_TRAIN_IMAGES = 16 * 798

NUM_VAL_IMAGES = 16 * 232
AUTOTUNE = tf.data.experimental.AUTOTUNE



@tf.function

def read_tfrec(tfrec):

        feature = {

            'image': tf.io.FixedLenFeature([], tf.string),

            'class': tf.io.FixedLenFeature([], tf.int64)

        }

        example = tf.io.parse_single_example(tfrec, feature)

        return example['image'], example['class']

    

@tf.function

def get_image_and_class(image, classl):

    classl = tf.cast(classl, tf.float32)



    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    image = tf.cast(image, tf.float32)

    image = image / 255.

    

    return image, classl

    

def get_dataset(ds_path):

    cycle_length = 32

    shuffle_buffer = 1024

    

    ds = tf.data.Dataset.list_files(ds_path + '/*.tfrec')

    ds = ds.interleave(tf.data.TFRecordDataset, 

                       cycle_length=cycle_length, 

                       num_parallel_calls=AUTOTUNE)

    ds = ds.map(read_tfrec, 

                num_parallel_calls=AUTOTUNE)

    ds = ds.map(get_image_and_class,

               num_parallel_calls=AUTOTUNE)

    ds = ds.shuffle(shuffle_buffer)

    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE, 

                  drop_remainder=True)

    ds = ds.cache()

    ds = ds.prefetch(AUTOTUNE)

    return ds
tpu_cluster = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu_cluster)

tf.tpu.experimental.initialize_tpu_system(tpu_cluster)

strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster)
tf.keras.backend.clear_session()
EPOCHS = 20



LR_START = LR_MIN = 1e-5

LR_MAX = 5e-5 * strategy.num_replicas_in_sync



LR_WARMUP_EPOCHS = 5

LR_EXP_DECAY = 0.8



def get_lr(epoch):

    if epoch < LR_WARMUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_WARMUP_EPOCHS * epoch + LR_START

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_WARMUP_EPOCHS) + LR_MIN

    return lr



cbs = [tf.keras.callbacks.LearningRateScheduler(get_lr, verbose=True)]
with strategy.scope():

    effnet = kapps.efficientnet.EfficientNetB7(include_top=False, 

                                               weights='imagenet', 

                                               pooling='avg')

    effnet.trainable = True

    

    model = tf.keras.models.Sequential([

        effnet,

        tf.keras.layers.Dense(NUM_CLASSES, 

                              name='preds', 

                              activation='softmax')

    ], name='finetune_effnet_b7')

    model.summary()

    

    model.compile(loss='sparse_categorical_crossentropy', 

                  optimizer='adam', 

                  metrics=['accuracy'])
train_ds = get_dataset(TRAIN_PATH)

val_ds = get_dataset(VAL_PATH)



history = model.fit(train_ds, 

                    steps_per_epoch=NUM_TRAIN_IMAGES // BATCH_SIZE, 

                    validation_data=val_ds, 

                    validation_steps=NUM_VAL_IMAGES // BATCH_SIZE,

                    epochs=EPOCHS, 

                    callbacks=cbs)
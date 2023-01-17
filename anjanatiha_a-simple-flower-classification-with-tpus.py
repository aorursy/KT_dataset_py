import tensorflow as tf

from kaggle_datasets import KaggleDatasets

import numpy as np



print("Tensorflow version " + tf.__version__)

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
GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"
NUM_TRAINING_IMAGES = 12753

NUM_TEST_IMAGES = 7382



IMG_SING_DIM = 224

IMG_DIM = str(IMG_SING_DIM) + 'x' + str(IMG_SING_DIM)



IMAGE_SIZE = [IMG_SING_DIM, IMG_SING_DIM] # at this size, a GPU will run out of memory. Use the TPU-+*



BATCH_SIZE = 16 * strategy.num_replicas_in_sync

def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['id']

    return image, idnum # returns a dataset of image(s)



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def data_augment(image, label):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_saturation(image, 0, 2)

    return image, label



def get_training_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-' + IMG_DIM + '/train/*.tfrec'), labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls = AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset



def get_validation_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-' + IMG_DIM + '/val/*.tfrec'), labeled=True, ordered=False)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-' + IMG_DIM + '/test/*.tfrec'), labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset





training_dataset = get_training_dataset()

validation_dataset = get_validation_dataset()
!pip install -q efficientnet
import efficientnet.tfkeras as efficientnet

from tensorflow.keras import optimizers, layers



with strategy.scope():    

    pretrained_model = efficientnet.EfficientNetB7(weights = 'noisy-student', include_top = False, input_shape = [*IMAGE_SIZE, 3])



#     pretrained_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    pretrained_model.trainable = True # tramsfer learning

    

    model = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(104, activation='softmax')

    ])

    



LEARNING_RATE = 3e-5 * strategy.num_replicas_in_sync

metric_list = ['sparse_categorical_accuracy']



optimizer = optimizers.Adam(lr=LEARNING_RATE)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=metric_list)

model.summary()

import seaborn as sns

import matplotlib.pyplot as plt



EPOCHS = 20

ES_PATIENCE = 6



LR_START = 0.00000001

LR_MIN = 0.000001

LR_MAX = LEARNING_RATE

LR_RAMPUP_EPOCHS = 3

LR_SUSTAIN_EPOCHS = 0

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 6))

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler





# checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True)

es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, 

                   restore_best_weights=True, verbose=1)

lr_callback = LearningRateScheduler(lrfn, verbose=1)



callback_list = [es, lr_callback]
# LEARNING_RATE = 3e-5 * strategy.num_replicas_in_sync



# optimizer = optimizers.Adam(lr=LEARNING_RATE)

# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=metric_list)







STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE



history = model.fit(x=get_training_dataset(), 

                    steps_per_epoch=STEPS_PER_EPOCH, 

                    validation_data=get_validation_dataset(),

                    callbacks=callback_list,

                    epochs=EPOCHS, 

                    verbose=2).history
def plot_metrics(history, metric_list):

    fig, axes = plt.subplots(len(metric_list), 1, sharex='col', figsize=(24, 12))

    axes = axes.flatten()

    

    for index, metric in enumerate(metric_list):

        axes[index].plot(history[metric], label='Train %s' % metric)

        axes[index].plot(history['val_%s' % metric], label='Validation %s' % metric)

        axes[index].legend(loc='best', fontsize=16)

        axes[index].set_title(metric)



    plt.xlabel('Epochs', fontsize=16)

    sns.despine()

    plt.show()



plot_metrics(history, metric_list=['loss', 'sparse_categorical_accuracy'])
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)



print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
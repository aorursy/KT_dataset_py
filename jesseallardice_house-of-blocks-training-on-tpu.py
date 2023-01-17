import math, re, os

import cv2

import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping



print(f"Tensor Flow version: {tf.__version__}")

AUTO = tf.data.experimental.AUTOTUNE
try: # detect TPUs

#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except ValueError: # detect GPU

    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines

    # strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU

    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines



print(f"Number of Accelerators: {strategy.num_replicas_in_sync}")
GCS_PATH = KaggleDatasets().get_gcs_path('tfrecords-for-adl-wustl-fall-2020')
!gsutil ls $GCS_PATH
EPOCHS = 24

IMAGE_SIZE = [192,192]



BLOCKS_TRAIN_DATASETS = { # avialable image sizes

    192: GCS_PATH + '/Train/192x192/*.tfrec',

    331: GCS_PATH + '/Train/331x331/*.tfrec',

}

CLASSES = [0,1]

assert IMAGE_SIZE[0] == IMAGE_SIZE[1], "only square images are supported"

assert IMAGE_SIZE[0] in BLOCKS_TRAIN_DATASETS, "this image size is not supported"



# learning rate schedule for TPU, GPU and CPU.

# using a LR ramp up because fine-tuning a pre-trained model.

# startin with a high LR would break the pre-trained weights.



BATCH_SIZE = 16 * strategy.num_replicas_in_sync # this is 8 on TPU v3-8, it is 1 on CPU and GPU

LR_START = 0.00001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 5

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



lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng,y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
def dataset_to_numpy_util(dataset, N):

    dataset = dataset.unbatch().batch(N)

    for images, labels in dataset:

        numpy_images = images.numpy()

        numpy_labels = labels.numpy()

        break;  

    return numpy_images, numpy_labels



def title_from_label_and_target(label, correct_label):

    if correct_label is None:

        return CLASSES[label], True

    correct = (label == correct_label)

    return "{} [{}{}{}]".format(

        CLASSES[int(label)], 

        'OK' if correct else 'NO', 

        u"\u2192" if not correct else '',

        CLASSES[correct_label] if not correct else ''

    ), correct



def display_one_flower(image, title, subplot, red=False):

    plt.subplot(subplot)

    plt.axis('off')

    plt.imshow(image)

    plt.title(title, fontsize=16, color='red' if red else 'black')

    return subplot+1

  

def display_9_images_from_dataset(dataset):

    subplot=331

    plt.figure(figsize=(13,13))

    images, labels = dataset_to_numpy_util(dataset, 9)

    for i, image in enumerate(images):

        title = CLASSES[labels[i]]

        subplot = display_one_flower(image, title, subplot)

        if i >= 8:

            break;

              

    #plt.tight_layout()

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.show()  



def display_9_images_with_predictions(images, predictions, labels):

    subplot=331

    plt.figure(figsize=(13,13))

    for i, image in enumerate(images):

        title, correct = title_from_label_and_target(predictions[i], labels[i])

        subplot = display_one_flower(image, title, subplot, not correct)

        if i >= 8:

            break;

              

    #plt.tight_layout()

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.show()

    

def display_training_curves(training, validation, title, subplot):

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

        #plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    #ax.set_ylim(0.28,1.05)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])
def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



gcs_pattern = BLOCKS_TRAIN_DATASETS[IMAGE_SIZE[0]]

validation_split = 0.20

filenames = tf.io.gfile.glob(gcs_pattern)

split = len(filenames) - int(len(filenames) * validation_split)

TRAINING_FILENAMES = filenames[:split]

VALIDATION_FILENAMES = filenames[split:]

TRAIN_STEPS = count_data_items(TRAINING_FILENAMES) // BATCH_SIZE

print("TRAINING IAGES; ", count_data_items(TRAINING_FILENAMES), ", STEPS PER EPOCH: ", TRAIN_STEPS)

print("VALIDATION IMAGES: ", count_data_items(VALIDATION_FILENAMES))



def read_tfrecord(example):

    features = {

        "image": tf.io.FixedLenFeature([],tf.string),

        "id": tf.io.FixedLenFeature([], tf.int64),

        "filename": tf.io.FixedLenFeature([],tf.string),

        "stable": tf.io.FixedLenFeature([], tf.int64),

    }

    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example["image"], channels=3)

    image = tf.cast(image,tf.float32) / 255.0 # convert image to floats in [0, 1] range

    target = tf.cast(example["stable"],tf.int32)

    return image, target



def force_image_sizes(dataset, image_size):

    # explicit size need for TPU

    reshape_images = lambda image, label: (tf.reshape(image, [*image_size, 3]), label)

    dataset = dataset.map(reshape_images, num_parallel_calls=AUTO)

    return dataset



def load_dataset(filenames):

    # read from TFRecords. For optimal performance, reading from multiplie files at once

    # and disregarding data order. Order does not matter since we will suffle the data away.

    

    ignore_order = tf.data.Options()

    ignore_order.experimental_deterministic = False

    

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.with_options(ignore_order)

    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

    dataset = force_image_sizes(dataset, IMAGE_SIZE)

    return dataset



def data_augment(image, target):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    # image = tf.image.random_saturation(image, 0, 2)

    # random brightness/exposure?

    return image, target 



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset():

    dataset = load_dataset(VALIDATION_FILENAMES)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
training_dataset = get_training_dataset()

validation_dataset = get_validation_dataset()
display_9_images_from_dataset(validation_dataset)
def create_model():

    #pretrained_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)

    pretrained_model = tf.keras.applications.Xception(input_shape=[*IMAGE_SIZE, 3], include_top=False)

    #pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])

    #pretrained_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])

    # EfficientNet can be loaded through efficientnet.tfkeras library (https://github.com/qubvel/efficientnet)

    #pretrained_model = efficientnet.tfkeras.EfficientNetB0(weights='imagenet', include_top=False)

    

    pretrained_model.trainable = True



    model = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        #tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(1, activation='sigmoid')

    ])



    model.compile(

        optimizer='adam',

        loss = 'binary_crossentropy',

        metrics=['accuracy']

    )



    return model
with strategy.scope():

    model = create_model()

model.summary()
es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1, mode='auto',

        restore_best_weights=True)



callback_list = [es, lr_callback]



history = model.fit(training_dataset, validation_data=validation_dataset,

                    steps_per_epoch=TRAIN_STEPS, epochs=EPOCHS, callbacks=callback_list)



final_accuracy = history.history["val_accuracy"][-1:]

print("FINAL ACCURACY MEAN-1: ", np.mean(final_accuracy))
display_training_curves(history.history['accuracy'][0:], history.history['val_accuracy'][0:], 'accuracy', 211)

display_training_curves(history.history['loss'][0:], history.history['val_loss'][0:], 'loss', 212)
# # a couple of images to test predictions too

# some_flowers, some_labels = dataset_to_numpy_util(validation_dataset, 160)
# # randomize the input so that you can execute multiple times to change results

# permutation = np.random.permutation(8*20)

# some_flowers, some_labels = (some_flowers[permutation], some_labels[permutation])



# predictions = model.predict(some_flowers, batch_size=16)

# evaluations = model.evaluate(some_flowers, some_labels, batch_size=16)

  

# print(predictions.tolist())

# print('[val_loss, val_acc]', evaluations)



# display_9_images_with_predictions(some_flowers, predictions, some_labels)
# # TPUs need this extra setting to save to local disk, otherwise, they can only save models to GCS (Google Cloud Storage).

# # The setting instructs Tensorflow to retrieve all parameters from the TPU then do the saving from the local VM, not the TPU.

# save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

# model.save('./model', options=save_locally) # saving in Tensorflow's "saved model" format
model.save('model.h5')
BLOCKS_TEST_DATASETS = { # avialable image sizes

    192: GCS_PATH + '/Test/192x192/*.tfrec',

    331: GCS_PATH + '/Test/331x331/*.tfrec',

}

CLASSES = [0,1]

assert IMAGE_SIZE[0] == IMAGE_SIZE[1], "only square images are supported"

assert IMAGE_SIZE[0] in BLOCKS_TEST_DATASETS, "this image size is not supported"
gcs_pattern = BLOCKS_TEST_DATASETS[IMAGE_SIZE[0]]





TEST_FILENAMES = tf.io.gfile.glob(gcs_pattern)

print("TEST IMAGES; ", count_data_items(TEST_FILENAMES))
def read_test_tfrecord(example):

    features = {

        "image": tf.io.FixedLenFeature([],tf.string),

        "id": tf.io.FixedLenFeature([], tf.int64),

        "filename": tf.io.FixedLenFeature([],tf.string),

    }

    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example["image"], channels=3)

    image = tf.cast(image,tf.float32) / 255.0 # convert image to floats in [0, 1] range

    return image



def force_test_image_sizes(dataset, image_size):

    # explicit size need for TPU

    reshape_images = lambda image: tf.reshape(image, [*image_size, 3])

    dataset = dataset.map(reshape_images, num_parallel_calls=AUTO)

    return dataset



def load_test_dataset(filenames):

    # read from TFRecords. For optimal performance, reading from multiplie files at once

    # and disregarding data order. Order does not matter since we will suffle the data away.

    

    ignore_order = tf.data.Options()

    ignore_order.experimental_deterministic = True # want in order

    

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.with_options(ignore_order)

    dataset = dataset.map(read_test_tfrecord, num_parallel_calls=AUTO)

    dataset = force_test_image_sizes(dataset, IMAGE_SIZE)

    return dataset



def get_test_dataset():

    dataset = load_test_dataset(TEST_FILENAMES)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
test_dataset = get_test_dataset()

y_pred = model.predict(test_dataset, batch_size=16)

y_pred
def test_dataset_to_numpy_util(dataset): #, N):

    dataset = dataset #.unbatch() #.batch(N)

    for IDs in dataset:

        numpy_IDs = IDs.numpy()

        break;  

    return numpy_IDs



def read_test_tfrecord_id(example):

    features = {

        "image": tf.io.FixedLenFeature([],tf.string),

        "id": tf.io.FixedLenFeature([], tf.int64),

        "filename": tf.io.FixedLenFeature([],tf.string),

    }

    example = tf.io.parse_single_example(example, features)

    ID = example["id"]

    return ID



def load_test_id(filenames):

    ignore_order = tf.data.Options()

    ignore_order.experimental_deterministic = True # want in order

    

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.with_options(ignore_order)

    dataset = dataset.map(read_test_tfrecord_id, num_parallel_calls=AUTO)

    return dataset

    

def get_test_id():

    dataset = load_test_id(TEST_FILENAMES)

    return dataset

test_id_dataset = get_test_id()

test_id = test_dataset_to_numpy_util(test_id_dataset)

x = [IDs.numpy() for IDs in test_id_dataset]
PATH = "/kaggle/input/applications-of-deep-learning-wustl-fall-2020/final-kaggle-data/"



PATH_TEST = os.path.join(PATH, "test.csv")



df_test = pd.read_csv(PATH_TEST)



df_test.info()
df_submit = pd.DataFrame({"id":np.array(x).flatten(), "stable":y_pred.flatten()})

df_submit = df_test.merge(df_submit, how='inner')

df_submit.head()
df_submit.info()
df_submit.to_csv("/kaggle/working/submit.csv",index = False)
!pip install efficientnet
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
import os, re



import numpy as np



import matplotlib.pyplot as plt



import tensorflow as tf

import efficientnet.tfkeras as enet

from kaggle_datasets import KaggleDatasets

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zca_whitening=True)



from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import os, sys, math

AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

#tf.debugging.set_log_device_placement(True)

# TPU HARDWARE DETECT



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
# TRAIN_IMG_PATH = '../input/flower-classification-with-tpus/tfrecords-jpeg-192x192/train/' 

# VALID_IMG_PATH = '../input/flower-classification-with-tpus/tfrecords-jpeg-192x192/val/' 

# TEST_IMG_PATH = '../input/flower-classification-with-tpus/tfrecords-jpeg-192x192/test/'





# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()



# Configuration

TARGET_SIZE = [512, 512]



GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}

GCS_PATH = GCS_PATH_SELECT[TARGET_SIZE[0]]



TRAIN_IMG_PATH = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

VALID_IMG_PATH = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_IMG_PATH = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for 



EPOCHS = 45

BATCH_SIZE = 16 * strategy.num_replicas_in_sync
# CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily', 

#            'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09

#            'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower', 

#            'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19

#            'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy', 

#            'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29

#            'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya',

#            'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39

#            'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',   

#            'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49

#            'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff', 

#            'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59

#            'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',    

#            'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69

#            'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',    

#            'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79

#            'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',      

#            'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89

#            'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',     

#            'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99

#            'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']
# tf-record file read

def tfrecord_fn(record):

    columns = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "class": tf.io.FixedLenFeature([], tf.int64)

    }

    

    # decode the tfrecord

    example = tf.io.parse_single_example(record, columns)

    image = tf.image.decode_jpeg(example['image'], channels=3)

    image = tf.reshape(image, [*TARGET_SIZE, 3])

    label = tf.cast(example['class'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



    
# set experimental_deterministic = False to read from multiple files



option_no_order = tf.data.Options()

option_no_order.experimental_deterministic = False



filenames = tf.io.gfile.glob(TRAIN_IMG_PATH)

dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

dataset = dataset.with_options(option_no_order)

dataset = dataset.map(tfrecord_fn, num_parallel_calls=AUTO)

dataset = dataset.shuffle(2048)
for image, label in dataset.take(4):

    print(label.numpy())
plt.figure(figsize=(15,15))

subCount = 1   # # plot number

rowCount =3      # No of images in row

colCount =4      # No of images in columns



for i, (image, label) in enumerate(dataset):

    plt.subplot(rowCount, colCount, subCount)

    plt.axis('off')

    plt.imshow(image.numpy().astype(np.uint8))

    plt.title(label.numpy(), fontsize=16)

    subCount = subCount + 1

    if i ==11:           # (row*column)-1

        break

plt.tight_layout()

plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.show()
# tf-record label file read

def tfrecord_label_fn(record):

    columns = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "class": tf.io.FixedLenFeature([], tf.int64)

    }

    

    # decode the tfrecord

    example = tf.io.parse_single_example(record, columns)

    image = tf.image.decode_jpeg(example['image'], channels=3)

    

    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # 0-1

    

    #image = tf.cast(image, tf.float32)# //255.0            # supported data type tf.float32, tf.int32, tf.bfloat16 

    image = tf.reshape(image, [*TARGET_SIZE, 3])

    #image = datagen.fit(image)

    label = tf.cast(example['class'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs
# tf-record unlabelled file read

def tfrecord_unlabel_fn(record):

    columns = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "id": tf.io.FixedLenFeature([], tf.string)

    }

    

    # decode the tfrecord

    example = tf.io.parse_single_example(record, columns)

    image = tf.image.decode_jpeg(example['image'], channels=3)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # 0-1

    

    #image = tf.cast(image, tf.float32)# //255.0                  # supported data type tf.float32, tf.int32, tf.bfloat16 

    image = tf.reshape(image, [*TARGET_SIZE, 3])

    # fit parameters from data

#    image = datagen.fit(image)

    idn = example['id']

    return image, idn # returns a dataset of (image, id) pairs



    
def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    #image = tf.image.random_saturation(image, 0, 2)

    return image, label   
def training_data_fn():

    # set experimental_deterministic = False to read from multiple files



    option_no_order = tf.data.Options()

    option_no_order.experimental_deterministic = False



    filenames = tf.io.gfile.glob(TRAIN_IMG_PATH)

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.with_options(option_no_order)

    dataset = dataset.map(tfrecord_label_fn, num_parallel_calls=AUTO)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
def validation_data_fn():

    # set experimental_deterministic = False to read from multiple files



    option_no_order = tf.data.Options()

    option_no_order.experimental_deterministic = False



    filenames = tf.io.gfile.glob(VALID_IMG_PATH)

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.with_options(option_no_order)

    dataset = dataset.map(tfrecord_label_fn, num_parallel_calls=AUTO)

    #dataset = dataset.repeat() # the training dataset must repeat for several epochs

    #dataset = dataset.shuffle(2048)

    dataset = dataset.cache()

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
def test_data_fn():

    # set experimental_deterministic = False to read from multiple files 



    option_no_order = tf.data.Options()

    option_no_order.experimental_deterministic = False



    filenames = tf.io.gfile.glob(TEST_IMG_PATH)

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.with_options(option_no_order)

    dataset = dataset.map(tfrecord_unlabel_fn, num_parallel_calls=AUTO)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
# Plot function to diaplay the images of the dataset

def plot_view_fn(dataset, num_row, num_col):

    plt.figure(figsize=(13,13))

    subCount = 1   # # plot number

    rowCount =num_row      # No of images in row

    colCount =num_col      # No of images in columns



    for i, (image, label) in enumerate(dataset):

        plt.subplot(rowCount, colCount, subCount)

        plt.axis('off')

        plt.imshow(image.numpy().astype(np.uint8)) 

        plt.title(label.numpy(), fontsize=16)

        subCount = subCount + 1

        if i ==(rowCount*colCount)-1:           # (row*column)-1

            break

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.show()
# # Plot function to diaplay the images of the dataset

# def plot_view_fn(dataset, num_row, num_col, label=True):

#     plt.figure(figsize=(13,13))

#     subCount = 1   # # plot number

#     rowCount =num_row      # No of images in row

#     colCount =num_col      # No of images in columns



#     for i, (image, label) in enumerate(dataset):

#         plt.subplot(rowCount, colCount, subCount)

#         plt.axis('off')

#         plt.imshow(image.numpy().astype(np.uint8))

#         if label is False:

#             plt.title('')

#         else:

#             plt.title(label.numpy(), fontsize=16)

#         subCount = subCount + 1

#         if i ==(rowCount*colCount)-1:           # (row*column)-1

#             break

#     plt.tight_layout()

#     plt.subplots_adjust(wspace=0.1, hspace=0.1)

#     plt.show()
# # View the training Dataset

# training_dataset = training_data_fn()

# training_dataset = training_dataset.unbatch()

# train_batch = iter(training_dataset)
# plot_view_fn(train_batch, 3, 4)
# # View the validation Dataset

# validation_dataset = validation_data_fn()

# validation_dataset = validation_dataset.unbatch()

# valid_batch = iter(validation_dataset)
# plot_view_fn(valid_batch, 4, 4)
# # View the test Dataset

# test_dataset = test_data_fn()

# test_dataset = test_dataset.unbatch()

# test_batch = iter(test_dataset)
# plot_view_fn(test_batch, 4, 4)
def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAIN_IMG_PATH)

NUM_VALIDATION_IMAGES = count_data_items(VALID_IMG_PATH)

NUM_TEST_IMAGES = count_data_items(TEST_IMG_PATH)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
# For TPU TRAINING



with strategy.scope():

# EfficientNetB7

    enet = enet.EfficientNetB7(

        input_shape=(512, 512, 3),

        weights='imagenet',

        include_top=False

    )

    enet.trainable = True



    model = tf.keras.Sequential([

        enet,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(104, activation='softmax')

    ])
# For GPU TRAINING



# with tf.device('/GPU:0'): 

    

#     # EfficientNetB7



#     enet = enet.EfficientNetB7(

#             input_shape=(192, 192, 3),

#             weights='imagenet',

#             include_top=False

#         )



#     model = tf.keras.Sequential([

#             enet,

#             tf.keras.layers.GlobalAveragePooling2D(),

#             tf.keras.layers.Dense(104, activation='softmax')

#         ])





    

    
# # EfficientNetB7



# enet = enet.EfficientNetB7(

#         input_shape=(192, 192, 3),

#         weights='imagenet',

#         include_top=False

#     )



# model = tf.keras.Sequential([

#         enet,

#         tf.keras.layers.GlobalAveragePooling2D(),

#         tf.keras.layers.Dense(104, activation='softmax')

#     ])



model.compile(

    optimizer=tf.keras.optimizers.Adam(learning_rate=1.5e-05, beta_1=0.9, beta_2=0.99, amsgrad=False),

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)

model.summary()
#model.get_layer('dense').kernel_regularizer = tf.keras.regularizers.l2(0.0001) 
# model.compile(

#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99, amsgrad=False),

#     loss = 'sparse_categorical_crossentropy',

#     metrics=['sparse_categorical_accuracy']

# )

# model.summary()
# def lr_fn(epoch):

#     LR_START = 0.00001

#     LR_MAX = 0.00005 * strategy.num_replicas_in_sync

#     LR_MIN = 0.00001

#     LR_RAMPUP_EPOCHS = 10

#     LR_SUSTAIN_EPOCHS = 0

#     LR_EXP_DECAY = .8

    

#     if epoch < LR_RAMPUP_EPOCHS:

#         lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

#     elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

#         lr = LR_MAX

#     else:

#         lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

#     return lr

# lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_fn, verbose=True)

# rng = [i for i in range(EPOCHS)]

# y = [lr_fn(x) for x in rng]

# plt.plot(rng, y)

# print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_fn, verbose=1)

# Early stopping

callback_stop = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=5, verbose=1, mode='auto', restore_best_weights=True)
history = model.fit(

    training_data_fn(),

    steps_per_epoch=STEPS_PER_EPOCH,

    epochs=EPOCHS, 

    callbacks=[callback_stop],   #callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',

    validation_data=validation_data_fn()

)
plt.plot(history.history['loss'], label='train loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.xlabel("epoch")

plt.ylabel("Cross-entropy loss")

plt.legend();

plt.plot(history.history['sparse_categorical_accuracy'], label='train accuracy')

plt.plot(history.history['val_sparse_categorical_accuracy'], label='val accuracy')

plt.xlabel("epoch")

plt.ylabel("categorical_accuracy")

plt.legend();

cmdataset = validation_data_fn()

images_ds = cmdataset.map(lambda image, label: image)

labels_ds = cmdataset.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

cm_probabilities = model.predict(images_ds)

cm_predictions = np.argmax(cm_probabilities, axis=-1)

print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)

print("Predicted labels: ", cm_predictions.shape, cm_predictions)
score = f1_score(cm_correct_labels, cm_predictions, labels=range(104), average='macro')

precision = precision_score(cm_correct_labels, cm_predictions, labels=range(104), average='macro')

recall = recall_score(cm_correct_labels, cm_predictions, labels=range(104), average='macro')

print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
test_ds = test_data_fn()



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idn: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)



print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idn: idn).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
# from keras import backend as K



# def recall_m(y_true, y_pred):

#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

#     recall = true_positives / (possible_positives + K.epsilon())

#     return recall



# def precision_m(y_true, y_pred):

#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

#     precision = true_positives / (predicted_positives + K.epsilon())

#     return precision



# def f1_m(y_true, y_pred):

#     precision = precision_m(y_true, y_pred)

#     recall = recall_m(y_true, y_pred)

#     return 2*((precision*recall)/(precision+recall+K.epsilon()))







# model.compile(

#     optimizer=tf.keras.optimizers.Adam(learning_rate=1.5e-05, beta_1=0.9, beta_2=0.99, amsgrad=False),

#     loss = 'sparse_categorical_crossentropy',

#     metrics=['acc',f1_m]#metrics=['sparse_categorical_accuracy'] metrics=['acc',f1_m,precision_m, recall_m]

# )

# model.summary()
# history01 = model.fit(

#     training_data_fn(),

#     steps_per_epoch=STEPS_PER_EPOCH,

#     epochs=EPOCHS, 

#     callbacks=[callback_stop],   #callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',

#     validation_data=validation_data_fn()

# )
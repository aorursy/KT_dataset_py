!pip install -q efficientnet
import math, re, os

import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

import efficientnet.tfkeras as efn

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from tensorflow.keras.applications import InceptionResNetV2

from tensorflow.keras.applications import ResNet152V2

from tensorflow.keras.applications import DenseNet201

from tensorflow.keras.applications import VGG19

from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
#使用TPU进行训练

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

#GCS_DS_PATH = 'H:/software'
#根据需求选择数据集，本例进行TPU进行训练，使用512*512

IMAGE_SIZE = [512, 512]

EPOCHS = 20

#BATCH_SIZE = 16

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



GCS_PATH_SELECT = {

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}



GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FIFENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

VALIDATION_FIFENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FIFENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')





def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



#labeled tfrecord 是train、validation 数据，用tf.io.parse_single_example读入TFRecord数据



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), 

        "class": tf.io.FixedLenFeature([], tf.int64),  

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = example['image']

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    label = tf.cast(example['class'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = example['image']

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    label = example['id']

    return image, label # returns a dataset of (image, label) pairs



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    #image = tf.image.random_saturation(image, 0, 2)

    return image, label



ignore_order = tf.data.Options()



dataset_train = tf.data.TFRecordDataset(TRAINING_FIFENAMES)

dataset_train = dataset_train.with_options(ignore_order)

dataset_train = dataset_train.map(read_labeled_tfrecord)   #map居然能通过，而直接使用函数缺不可以？

dataset_train = dataset_train.map(data_augment, num_parallel_calls=AUTO)

dataset_train = dataset_train.repeat() #

dataset_train = dataset_train.shuffle(2048)

dataset_train = dataset_train.batch(BATCH_SIZE)

dataset_train = dataset_train.prefetch(AUTO)



dataset_val = tf.data.TFRecordDataset(VALIDATION_FIFENAMES)

dataset_val = dataset_val.with_options(ignore_order)

dataset_val = dataset_val.map(read_labeled_tfrecord)   #map居然能通过，而直接使用函数缺不可以？

dataset_val = dataset_val.map(data_augment, num_parallel_calls=AUTO)

dataset_val = dataset_val.batch(BATCH_SIZE)

dataset_val = dataset_val.cache()

dataset_val = dataset_val.prefetch(AUTO)



dataset_test = tf.data.TFRecordDataset(TEST_FIFENAMES)

dataset_test = dataset_test.with_options(ignore_order)

dataset_test = dataset_test.map(read_unlabeled_tfrecord)   #map居然能通过，而直接使用函数缺不可以？

dataset_test = dataset_test.map(data_augment, num_parallel_calls=AUTO)

dataset_test = dataset_test.batch(BATCH_SIZE)

dataset_test = dataset_test.prefetch(AUTO)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FIFENAMES)

NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FIFENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FIFENAMES)



STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE



CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09

           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19

           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29

           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39

           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49

           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59

           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69

           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79

           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89

           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99

           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102
def lr_schedule(epoch):

    # Learning Rate Schedule



    lr =1e-4

    total_epochs =epoch



    check_1 = int(total_epochs * 0.9)

    check_2 = int(total_epochs * 0.8)

    check_3 = int(total_epochs * 0.6)

    check_4 = int(total_epochs * 0.4)



    if epoch > check_1:

        lr *= 1e-1

    elif epoch > check_2:

        lr *= 1e-2

    elif epoch > check_3:

        lr *= 1e-3

    elif epoch > check_4:

        lr *= 1e-4



    return lr







lr_scheduler =tf.keras.callbacks.LearningRateScheduler(lr_schedule)



#lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)

with strategy.scope():

#     pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

#     pretrained_model.trainable = True # tramsfer learning

    

    enet = efn.EfficientNetB7(

        input_shape=[*IMAGE_SIZE, 3],

        weights='imagenet',

        include_top=False)

    Incep = InceptionResNetV2(

        input_shape=[*IMAGE_SIZE, 3],

        weights='imagenet',

        include_top=False)   

    res = ResNet152V2(

        input_shape=[*IMAGE_SIZE, 3],

        weights='imagenet',

        include_top=False)

    den = DenseNet201(

        input_shape=(512, 512, 3),

        weights='imagenet',

        include_top=False)

    

    

    model_1 = tf.keras.Sequential([

        enet,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

    

    model_2 = tf.keras.Sequential([

        Incep,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

    model_3 = tf.keras.Sequential([

        res,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

    model_4 = tf.keras.Sequential([

        den,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

  

          

model_1.compile(

    optimizer='adam',

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)

model_1.summary()



          

model_2.compile(

    optimizer='adam',

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)

model_2.summary()



          

model_3.compile(

    optimizer='adam',

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)

model_3.summary()



          

model_4.compile(

    optimizer='adam',

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)

model_4.summary()



history_1 = model_1.fit(dataset_train, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_scheduler],validation_data=dataset_val)

history_2 = model_2.fit(dataset_train, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_scheduler],validation_data=dataset_val)

history_3 = model_3.fit(dataset_train, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_scheduler],validation_data=dataset_val)

history_4 = model_4.fit(dataset_train, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_scheduler],validation_data=dataset_val)
def display_training_curves(training, validation, title, subplot):

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    #ax.set_ylim(0.28,1.05)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.']) 

    

display_training_curves(history_1.history['loss'], history_1.history['val_loss'], 'loss', 211)

display_training_curves(history_1.history['sparse_categorical_accuracy'], history_1.history['val_sparse_categorical_accuracy'], 'accuracy', 212)



display_training_curves(history_2.history['loss'], history_2.history['val_loss'], 'loss', 211)

display_training_curves(history_2.history['sparse_categorical_accuracy'], history_2.history['val_sparse_categorical_accuracy'], 'accuracy', 212)



display_training_curves(history_3.history['loss'], history_3.history['val_loss'], 'loss', 211)

display_training_curves(history_3.history['sparse_categorical_accuracy'], history_3.history['val_sparse_categorical_accuracy'], 'accuracy', 212)



display_training_curves(history_4.history['loss'], history_4.history['val_loss'], 'loss', 211)

display_training_curves(history_4.history['sparse_categorical_accuracy'], history_4.history['val_sparse_categorical_accuracy'], 'accuracy', 212)
dataset_val = tf.data.TFRecordDataset(VALIDATION_FIFENAMES)

# dataset_val = dataset_val.with_options(ignore_order)

dataset_val = dataset_val.map(read_labeled_tfrecord)   #map居然能通过，而直接使用函数缺不可以？

dataset_val = dataset_val.map(data_augment, num_parallel_calls=AUTO)

dataset_val = dataset_val.batch(BATCH_SIZE)

dataset_val = dataset_val.cache()

dataset_val = dataset_val.prefetch(AUTO)



images_ds = dataset_val.map(lambda image, label: image)

labels_ds = dataset_val.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() 

cm_probabilities = model.predict(images_ds)

cm_predictions = np.argmax(cm_probabilities, axis=-1)

print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)

print("Predicted labels: ", cm_predictions.shape, cm_predictions)
cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))

score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

#cmat = (cmat.T / cmat.sum(axis=1)).T # normalized

def display_confusion_matrix(cmat, score, precision, recall):

    plt.figure(figsize=(15,15))

    ax = plt.gca()

    ax.matshow(cmat, cmap='Reds')

    ax.set_xticks(range(len(CLASSES)))

    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    ax.set_yticks(range(len(CLASSES)))

    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})

    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    titlestring = ""

    if score is not None:

        titlestring += 'f1 = {:.3f} '.format(score)

    if precision is not None:

        titlestring += '\nprecision = {:.3f} '.format(precision)

    if recall is not None:

        titlestring += '\nrecall = {:.3f} '.format(recall)

    if len(titlestring) > 0:

        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})

    plt.show()

display_confusion_matrix(cmat, score, precision, recall)

print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
# dataset_test = tf.data.TFRecordDataset(TEST_FIFENAMES)

# # dataset_test = dataset_test.with_options(ignore_order)

# dataset_test = dataset_test.map(read_unlabeled_tfrecord)   #map居然能通过，而直接使用函数缺不可以？

# dataset_test = dataset_test.map(data_augment, num_parallel_calls=AUTO)

# dataset_test = dataset_test.batch(BATCH_SIZE)

# dataset_test = dataset_test.prefetch(AUTO)



# # test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



# print('Computing predictions...')

# test_images_ds = dataset_test.map(lambda image, idnum: image)

# probabilities = model.predict(test_images_ds)

# predictions = np.argmax(probabilities, axis=-1)

# print(predictions)



# print('Generating submission.csv file...')

# test_ids_ds = dataset_test.map(lambda image, idnum: idnum).unbatch()

# test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

# np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

# !head submission.csv
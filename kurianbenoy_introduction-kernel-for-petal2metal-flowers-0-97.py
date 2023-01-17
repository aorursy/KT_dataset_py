from IPython.display import YouTubeVideo

YouTubeVideo("JC84GCU7zqA")
from IPython.display import YouTubeVideo

YouTubeVideo("kBjYK3K3P6M")
!pip install  efficientnet



import efficientnet.tfkeras as efn

import re

import math

import numpy as np

import seaborn as sns



from kaggle_datasets import KaggleDatasets

from matplotlib import pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.metrics import TruePositives, FalsePositives, FalseNegatives

print("Tensorflow version " + tf.__version__)
# This is basically -1

AUTO = tf.data.experimental.AUTOTUNE

AUTO

# Cluster Resolver for Google Cloud TPUs.

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()



# Connects to the given cluster.

tf.config.experimental_connect_to_cluster(tpu)



# Initialize the TPU devices.

tf.tpu.experimental.initialize_tpu_system(tpu)



# TPU distribution strategy implementation.

strategy = tf.distribute.experimental.TPUStrategy(tpu)
# Configurations

IMAGE_SIZE = [512, 512]

EPOCHS = 30

BATCH_SIZE = 32 * strategy.num_replicas_in_sync

LEARNING_RATE = 1e-3

TTA_NUM = 5
print("Batch size used: ", BATCH_SIZE)
# As TPUs require access to the GCS path

GCS_DS_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started')

MORE_IMAGES_GCS_DS_PATH = KaggleDatasets().get_gcs_path('tf-flower-photo-tfrec')
GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition



MOREIMAGES_PATH_SELECT = {

    192: '/tfrecords-jpeg-192x192',

    224: '/tfrecords-jpeg-224x224',

    331: '/tfrecords-jpeg-331x331',

    512: '/tfrecords-jpeg-512x512'

}

MOREIMAGES_PATH = MOREIMAGES_PATH_SELECT[IMAGE_SIZE[0]]



IMAGENET_FILES = tf.io.gfile.glob(MORE_IMAGES_GCS_DS_PATH + '/imagenet' + MOREIMAGES_PATH + '/*.tfrec')

INATURELIST_FILES = tf.io.gfile.glob(MORE_IMAGES_GCS_DS_PATH + '/inaturalist' + MOREIMAGES_PATH + '/*.tfrec')

OPENIMAGE_FILES = tf.io.gfile.glob(MORE_IMAGES_GCS_DS_PATH + '/openimage' + MOREIMAGES_PATH + '/*.tfrec')
SKIP_VALIDATION = True



if SKIP_VALIDATION:

    TRAINING_FILENAMES = TRAINING_FILENAMES + VALIDATION_FILENAMES + IMAGENET_FILES + INATURELIST_FILES + OPENIMAGE_FILES



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

           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']               

print(f"No of Flower classes in dataset: {len(CLASSES)}")
# Learning rate schedule for TPU, GPU and CPU.

# Using an LR ramp up because fine-tuning a pre-trained model.

# Starting with a high LR would break the pre-trained weights.



LR_START = 0.00001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 5

LR_SUSTAIN_EPOCHS = 0

LR_EXP_DECAY = .7



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

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))



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

    

def display_training_curves(training, validation, title, subplot):

    with plt.xkcd():

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
def display_one_flower(image, title, subplot, red=False, titlesize=16):

    plt.subplot(*subplot)

    plt.axis('off')

    plt.imshow(image)

    if len(title) > 0:

        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))

    return (subplot[0], subplot[1], subplot[2]+1)



def display_batch_of_images(databatch, predictions=None):

    """This will work with:

    display_batch_of_images(images)

    display_batch_of_images(images, predictions)

    display_batch_of_images((images, labels))

    display_batch_of_images((images, labels), predictions)

    """

    # data

    images, labels = batch_to_numpy_images_and_labels(databatch)

    if labels is None:

        labels = [None for _ in enumerate(images)]

        

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle

    rows = int(math.sqrt(len(images)))

    cols = len(images)//rows

        

    # size and spacing

    FIGSIZE = 13.0

    SPACING = 0.1

    subplot=(rows,cols,1)

    if rows < cols:

        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))

    else:

        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    

    # display

    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):

        title = '' if label is None else CLASSES[label]

        correct = True

        if predictions is not None:

            title, correct = title_from_label_and_target(predictions[i], label)

        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images

        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)

    

    #layout

    plt.tight_layout()

    if label is None and predictions is None:

        plt.subplots_adjust(wspace=0, hspace=0)

    else:

        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    plt.show()

    

# Visualize model predictions

def dataset_to_numpy_util(dataset, N):

    dataset = dataset.unbatch().batch(N)

    for images, labels in dataset:

        numpy_images = images.numpy()

        numpy_labels = labels.numpy()

        break;  

    return numpy_images, numpy_labels



def title_from_label_and_target(label, correct_label):

    label = np.argmax(label, axis=-1)

    correct = (label == correct_label)

    return "{} [{}{}{}]".format(CLASSES[label], str(correct), ', shoud be ' if not correct else '',

                                CLASSES[correct_label] if not correct else ''), correct



def display_one_flower_eval(image, title, subplot, red=False):

    plt.subplot(subplot)

    plt.axis('off')

    plt.imshow(image)

    plt.title(title, fontsize=14, color='red' if red else 'black')

    return subplot+1



def display_9_images_with_predictions(images, predictions, labels):

    subplot=331

    plt.figure(figsize=(13,13))

    for i, image in enumerate(images):

        title, correct = title_from_label_and_target(predictions[i], labels[i])

        subplot = display_one_flower_eval(image, title, subplot, not correct)

        if i >= 8:

            break;

              

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.show()
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







def data_augment(image, label, seed=2020):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image, seed=seed)

#     image = tf.image.random_flip_up_down(image, seed=seed)

#     image = tf.image.random_brightness(image, 0.1, seed=seed)

    

#     image = tf.image.random_jpeg_quality(image, 85, 100, seed=seed)

#     image = tf.image.resize(image, [530, 530])

#     image = tf.image.random_crop(image, [512, 512], seed=seed)

    #image = tf.image.random_saturation(image, 0, 2)

    return image, label   



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_train_valid_datasets():

    dataset = load_dataset(TRAINING_FILENAMES + VALIDATION_FILENAMES, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
models = []

histories = []

# No of images in dataset

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = (NUM_TRAINING_IMAGES + NUM_VALIDATION_IMAGES) // BATCH_SIZE

print('Dataset: {} training images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES+NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def get_training_dataset_preview(ordered=True):

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO)

    return dataset



# Visualization utility functions

np.set_printoptions(threshold=15, linewidth=80)



def batch_to_numpy_images_and_labels(data):

    images, labels = data

    numpy_images = images.numpy()

    numpy_labels = labels.numpy()

    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings

        numpy_labels = [None for _ in enumerate(numpy_images)]

    # If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images, numpy_labels



def title_from_label_and_target(label, correct_label):

    if correct_label is None:

        return CLASSES[label], True

    correct = (label == correct_label)

    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',

                                CLASSES[correct_label] if not correct else ''), correct





# Visualize model predictions

def dataset_to_numpy_util(dataset, N):

    dataset = dataset.unbatch().batch(N)

    for images, labels in dataset:

        numpy_images = images.numpy()

        numpy_labels = labels.numpy()

        break;  

    return numpy_images, numpy_labels
train_dataset = get_training_dataset_preview(ordered=True)

y_train = next(iter(train_dataset.unbatch().map(lambda image, label: label).batch(NUM_TRAINING_IMAGES))).numpy()

print('Number of training images %d' % NUM_TRAINING_IMAGES)
display_batch_of_images(next(iter(train_dataset.unbatch().batch(20))))
# Label distribution

train_stack = np.asarray([[label, (y_train == index).sum()] for index, label in enumerate(CLASSES)])



fig, (ax1) = plt.subplots(1, 1, figsize=(24, 32))



ax1 = sns.barplot(x=train_stack[...,1], y=train_stack[...,0], order=CLASSES,ax=ax1)

ax1.set_title('Training labels', fontsize=30)

ax1.tick_params(labelsize=16)

# peer at test data

test_dataset = get_test_dataset()

test_dataset = test_dataset.unbatch().batch(20)

test_batch = iter(test_dataset)
# run this cell again for next set of images

display_batch_of_images(next(test_batch))
# def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

#     rotation = math.pi * rotation / 180.

#     shear = math.pi * shear/ 180.

    

#     c1 = tf.math.cos(rotation)

#     c2 = tf.math.sin(rotation)

#     one = tf.constant([1], dtype='float32')

#     zero = tf.constant([0], dtype='float32')

#     rotation_mat = tf.reshape(tf.concat([c1, s1, zero, -s1, c1, zero, \

#                                          zero, zero, one], axis=0), [3,3])

    

#     # shear matrix

#     c2 = tf.math.cos(shear)

#     s2 = tf.math.sin(shear)

#     shear_mat = tf.reshape(tf.concat([one, s2, zero, zero, c2, \

#                                          zero, zero, zero, one], axis=0), [3,3])

    

#     zoom_mat = tf.reshape(tf.concat([one/height_zoom, zero, zero, zero, \

#                                     oneb/width_zoom, zero, zero, zero, one], axis=0), [3,3])

    

#     shift_mat = tf.reshape(tf.concat([one_zero, height_shift, zero, one, width_shift, zero, \

#                                       zero, one], axis=0), [3,3])

    

#     return K.dot(K.dot(rotation_mat, shear_mat), K.dot(zoom_mat, shift_mat))
# def transform(image, label):

#     DIM = 512

#     XIM = DIM%2

    

#     rot = 15. * tf.random.normal([1], dtype='float32')

#     shr = 5.*tf.random.normal([1], dtype='float32')

#     h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 1.0

#     w_zoom = 1.0 + tf.random.normal([1], dtype='float32')/1.0

    

#     h_shift = 16. * tf.random.normal([1],dtype='float32') 

#     w_shift = 16. * tf.random.normal([1],dtype='float32') 

  

#     # GET TRANSFORMATION MATRIX

#     m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 



#     # LIST DESTINATION PIXEL INDICES

#     x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

#     y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

#     z = tf.ones([DIM*DIM],dtype='int32')

#     idx = tf.stack( [x,y,z] )

    

#     # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

#     idx2 = K.dot(m,tf.cast(idx,dtype='float32'))

#     idx2 = K.cast(idx2,dtype='int32')

#     idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

#     # FIND ORIGIN PIXEL VALUES           

#     idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

#     d = tf.gather_nd(image,tf.transpose(idx3))

        

#     return tf.reshape(d,[DIM,DIM,3]), label
# row = 3; col = 4;

# all_elements = get_training_dataset(load_dataset(TRAINING_FILENAMES),do_aug=False).unbatch()

# one_element = tf.data.Dataset.from_tensors( next(iter(all_elements)) )

# augmented_element = one_element.repeat().map(transform).batch(row*col)



# for (img,label) in augmented_element:

#     plt.figure(figsize=(15,int(15*row/col)))

#     for j in range(row*col):

#         plt.subplot(row,col,j+1)

#         plt.axis('off')

#         plt.imshow(img[j,])

#     plt.show()

#     break

# Need this line so Google will recite some incantations

# for Turing to magically load the model onto the TPU

with strategy.scope():

    enet = efn.EfficientNetB3(

        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

        weights='imagenet',

        include_top=False

    )

    

    enet.trainable = True

    model1 = tf.keras.Sequential([

        enet,

        tf.keras.layers.GlobalMaxPooling2D(name="Layer1"),

        tf.keras.layers.Dropout(0.),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

        

# METRICS = ['TruePositives','FalsePositives', 'FalseNegatives']

model1.compile(

    optimizer=tf.keras.optimizers.Adam(lr=0.0001),

    loss = 'sparse_categorical_crossentropy',

    metrics = "sparse_categorical_accuracy"

)



model1.summary()



models.append(model1)
# schedule = StepDecay(initAlpha=1e-4, factor=0.25, dropEvery=15)



# callbacks = [LearningRateScheduler(schedule)]
# Visualising the Model architecture

tf.keras.utils.plot_model(

    model1, to_file='model.png', show_shapes=True, show_layer_names=True,

)

%%time

Checkpoint=tf.keras.callbacks.ModelCheckpoint(f"Enet_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True,

       save_weights_only=True,mode='max')



train_history1 = model1.fit(

    get_training_dataset(), 

    steps_per_epoch=STEPS_PER_EPOCH,

    epochs=EPOCHS,

    callbacks=[lr_callback, Checkpoint, keras.callbacks.EarlyStopping(

        monitor="val_loss",

        min_delta=1e-2,

        patience=2,

        verbose=1,

    )],

)



histories.append(train_history1)



def plot_training(H):

	# construct a plot that plots and saves the training history

	with plt.xkcd():

		plt.figure()

		plt.plot(H.history["loss"], label="train_loss")

		plt.plot(H.history["sparse_categorical_accuracy"], label="train_accuracy")

		plt.title("Training Loss and Accuracy")

		plt.xlabel("Epoch #")

		plt.ylabel("Loss/Accuracy")

		plt.legend(loc="lower left")

		plt.show()
plot_training(train_history1)
with strategy.scope():

    densenet = tf.keras.applications.DenseNet201(input_shape=[*IMAGE_SIZE, 3], weights='imagenet', include_top=False)

    densenet.trainable = True

    

    model2 = tf.keras.Sequential([

        densenet,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

        

model2.compile(

    optimizer=tf.keras.optimizers.Adam(),

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)

model2.summary()
# Visualising the Model architecture

tf.keras.utils.plot_model(

    model1, to_file='model.png', show_shapes=True, show_layer_names=True,

)
%%time

Checkpoint=tf.keras.callbacks.ModelCheckpoint(f"Dnet_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True,

       save_weights_only=True,mode='max')

train_history2 = model2.fit(get_training_dataset(), 

                    steps_per_epoch=STEPS_PER_EPOCH,

                    epochs=15, 

                    callbacks = [lr_callback, Checkpoint, keras.callbacks.EarlyStopping(

        # Stop training when `val_loss` is no longer improving

        monitor="val_loss",

        # "no longer improving" being defined as "no better than 1e-2 less"

        min_delta=1e-2,

        # "no longer improving" being further defined as "for at least 2 epochs"

        patience=2,

        verbose=1,

    )])



histories.append(train_history2)
plot_training(train_history2)
if not SKIP_VALIDATION:

    cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.

    images_ds = cmdataset.map(lambda image, label: image)

    labels_ds = cmdataset.map(lambda image, label: label).unbatch()

    cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

    m1 = model1.predict(images_ds)

    m2 = model2.predict(images_ds)

    scores = []

    for alpha in np.linspace(0,1,100):

        cm_probabilities = alpha*m1+(1-alpha)*m2

        cm_predictions = np.argmax(cm_probabilities, axis=-1)

        scores.append(f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro'))



    best_alpha = np.argmax(scores)/100

else:

    best_alpha = 0.51  # change to value calculated with SKIP_VALIDATION=False

    

print('Best alpha: ' + str(best_alpha))
if not SKIP_VALIDATION:

    cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))

    score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

    precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

    recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

    #cmat = (cmat.T / cmat.sum(axis=1)).T # normalized

    display_confusion_matrix(cmat, score, precision, recall)

    print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
def predict_tta(model, n_iter):

    probs  = []

    for i in range(n_iter):

        test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.

        test_images_ds = test_ds.map(lambda image, idnum: image)

        probs.append(model.predict(test_images_ds,verbose=0))

        

    return probs
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Calculating predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probs1 = np.mean(predict_tta(model1, TTA_NUM), axis=0)

probs2 = np.mean(predict_tta(model2, TTA_NUM), axis=0)

probabilities = best_alpha*probs1 + (1-best_alpha)*probs2

predictions = np.argmax(probabilities, axis=-1)



print('Generating submission file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

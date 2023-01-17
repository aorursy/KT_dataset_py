import random, math, re, os
import tensorflow as tf, tensorflow.keras.backend as K
import numpy as np, pandas as pd
import h5py
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE
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
!ls /kaggle/input/
GCS_DS_PATH = KaggleDatasets().get_gcs_path()
IMAGE_SIZE = [224, 224] # At this size, a GPU will run out of memory. Use the TPU.
                        # For GPU training, please select 224 x 224 px image size.
EPOCHS = 30
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
FOLDS = 3
cross_validation=True

GCS_PATH_SELECT = { # available image sizes
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'
}
GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]

if cross_validation:
    TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec') + tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
    TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition
else:
    TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
    VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
    TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition

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
# numpy and matplotlib defaults
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

def display_one_flower(image, title, subplot, red=False, titlesize=16):
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)
    
def display_batch_of_images(databatch, predictions=None, convert=True):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    if convert: images, labels = batch_to_numpy_images_and_labels(databatch)
    else: images, labels = databatch
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
def flip_image_horizontally(image, label):
    flipped_image = tf.image.flip_left_right(image)
    return flipped_image, label

def flip_image_vertically(image, label):
    flipped_image = tf.image.flip_up_down(image)
    return flipped_image, label

def rotate_image(image, label, k=1):
    rotated_image = tf.image.rot90(image, k) # k=1 --> 90 degree, k=3 --> 270 degree
    return rotated_image, label

def zoom_in_image_20(image, label):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range = [0.8, 0.8]
    )
    #print(len(image))
    #print(image.shape())    
    #image_tensor = tf.convert_to_tensor(image.numpy())
    zoom_augmented = datagen.flow(image)
    zoom_image,label=next(zoom_augmented)
    return zoom_image, label

def zoom_in_image_30(image, label):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range = [0.7, 0.7]
    )
    zoom_augmented = datagen.flow(image.numpy(), label, shuffle=False, batch_size=1)
    zoom_image,label=next(zoom_augmented)
    return zoom_image, label

def zoom_in_image_40(image, label):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range = [0.6, 0.6]
    )
    zoom_augmented = datagen.flow(image.numpy(), label, shuffle=False, batch_size=1)
    zoom_image,label=next(zoom_augmented)
    return zoom_image, label

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

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def count_records(dataset):
    c = 0
    for record in dataset:
        c += 1
    return c

def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, 0, 2)
    return image, label   

def get_training_dataset(datasize, dataset, augment=False):
    if not cross_validation:
        dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    if not augment:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    else:
        h_flipped_dataset = dataset.map(flip_image_horizontally, num_parallel_calls=AUTO)
        v_flipped_dataset = dataset.map(flip_image_vertically, num_parallel_calls=AUTO)
        rotated_dataset = dataset.map(rotate_image, num_parallel_calls=AUTO)
        #zoomed_dataset_20percent = dataset.map(zoom_in_image_20, num_parallel_calls=AUTO)
        #zoomed_dataset_30percent = dataset.map(zoom_in_image_30, num_parallel_calls=AUTO)
        #zoomed_dataset_40percent = dataset.map(zoom_in_image_40, num_parallel_calls=AUTO)
        # concatenate all datasets
        dataset = dataset.concatenate(h_flipped_dataset)
        dataset = dataset.concatenate(v_flipped_dataset)
        dataset = dataset.concatenate(rotated_dataset)
        #dataset = dataset.concatenate(zoomed_dataset_20percent)
        #dataset = dataset.concatenate(zoomed_dataset_30percent)
        #dataset = dataset.concatenate(zoomed_dataset_40percent)        
        #datasize[0] = count_records(dataset)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs (to make infinite number of epochs)
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
    dataset = dataset.shuffle(2048) # shuffle the data by choosing random samples from a buffer of len 2048
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(dataset, ordered=False):
    if not cross_validation:
        dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
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

if cross_validation:
    NUM_TRAINING_IMAGES = int( count_data_items(TRAINING_FILENAMES) * (FOLDS-1.)/FOLDS )
    NUM_VALIDATION_IMAGES = int( count_data_items(TRAINING_FILENAMES) * (1./FOLDS) )
else:
    NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
    NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
train_data = load_dataset(TRAINING_FILENAMES)
#print(len(list(train_data.as_numpy_iterator()))
#d = train_data.batch(32)
#ite = iter(d)
#images, labels = next(ite)
#print(labels.numpy())

#dataset = train_data.map(data_augment, num_parallel_calls=AUTO)
print(train_data.take(1))
#print(len(list(dataset)))
'''
known_labels=[]
for image,label in train_data:
    #print(label.numpy())
    lab = label.numpy() 
    if lab not in known_labels:
        known_labels.append(lab)
known_labels.sort()
print("Known labels are {}".format(known_labels))
'''
print("Number of Classes = {}".format(len(CLASSES)))
tr = load_dataset(TRAINING_FILENAMES, ordered=True)
tr = tr.batch(2)
train_data_flipped = tr.map(flip_image_horizontally)
train_data_Vflipped = tr.map(flip_image_vertically)
train_data_rotated = tr.map(rotate_image)

train_batch = iter(tr)
train_batch_flipped = iter(train_data_flipped)
train_batch_Vflipped = iter(train_data_Vflipped)
train_batch_rotated = iter(train_data_rotated)

# https://keras.io/api/preprocessing/image/
"""
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#    rotation_range=90,
#    fill_mode="constant",
#    cval=0,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
    zoom_range = [0.6, 0.6]
#    horizontal_flip=True,
#    vertical_flip=True
  )

bat_size=1
batched_data = train_data.batch(bat_size)
batched_data = batched_data.prefetch(AUTO)
batch_iterator = iter(batched_data)

images,labels = next(batch_iterator)
#print(images)
print(labels)
#numpy_images = images.numpy()
#numpy_labels = labels.numpy()
#print(numpy_images)
#print(numpy_labels)


images_augmented = datagen.flow(images, labels, shuffle=False, batch_size=bat_size)
print(images_augmented)
display_batch_of_images((images,labels), convert=False)
print("After Zooming in")
image,label=next(images_augmented)
display_batch_of_images((image,label), convert=False)
"""
#display_batch_of_images(next(train_batch))
#print("After Horizontal Flipping:")
#display_batch_of_images(next(train_batch_flipped))
#print("After Vertical Flipping:")
#display_batch_of_images(next(train_batch_Vflipped))
#print("After Rotating:")
#display_batch_of_images(next(train_batch_rotated))
if not cross_validation:
    print("*** Training data shapes: ***")
    ds = [NUM_TRAINING_IMAGES]
    train_data = get_training_dataset(ds, augment=True)
    print("Number of training data after augmentation: {}".format(ds[0]))
    for image,label in train_data.take(1):
        print("Batch Size = {}".format(image.numpy().shape[0]))
        print("Images shape per batch = {0}, Labels shape per batch = {1}".format(image.numpy().shape, label.numpy().shape))

    print("*** Validation data shapes: ***")
    val_data = get_validation_dataset()
    for image,label in val_data.take(1):
        print("Batch Size = {}".format(image.numpy().shape[0]))
        print("Images shape per batch = {0}, Labels shape per batch = {1}".format(image.numpy().shape, label.numpy().shape))

    print("*** Test data shapes: ***")
    test_data = get_test_dataset()
    for image,idnum in test_data.take(1):
        print("Batch Size = {}".format(image.numpy().shape[0]))
        print("Images shape per batch = {0}, ID shape per batch = {1}".format(image.numpy().shape, label.numpy().shape))
# peek at the dataset
#train_batch = iter(train_data)
#test_batch = iter(test_data)
# run this cell again for next set of images
#display_batch_of_images(next(train_batch))
# run this cell again for next set of images
#display_batch_of_images(next(test_batch))
Training_strategies = ['Feature_Extractor','Partial_Fine_tuninig','Complete_Fine_Tuning']
train_strategy = Training_strategies[2]
print('train strategy is {}'.format(train_strategy))
if train_strategy == Training_strategies[0]:
    with strategy.scope():
        #pretrained_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        #pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        pretrained_model.trainable = False # False = transfer learning, True = fine-tuning
    
        model = tf.keras.Sequential([
            pretrained_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            #tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation='softmax')
        ])
        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

elif train_strategy == Training_strategies[1]:
    with strategy.scope():
        #pretrained_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        #pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        pretrained_model.trainable = True # False = transfer learning, True = fine-tuning
        
        for layer in pretrained_model.layers:
            if 'block5' in layer.name or 'block4' in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False
            print("{0} is trainable: {1}".format(layer.name, layer.trainable))
        
        model = tf.keras.Sequential([
            pretrained_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(len(CLASSES), activation='softmax')
        ])
        
        # optimizer
        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
        
elif train_strategy == Training_strategies[2]:
    with strategy.scope():
        pretrained_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        #pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        #pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        pretrained_model.trainable = True # False = transfer learning, True = fine-tuning
    
        model = tf.keras.Sequential([
            pretrained_model,
            #tf.keras.layers.Flatten(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation='softmax')
        ])
        # optimizer
        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

model.compile(
    optimizer=opt,
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)
model.summary()
if not cross_validation:
    EPOCHS=150   #15
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1, patience=14)
    mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
    history = model.fit(train_data, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=val_data, verbose=2, callbacks=[es, mc])
EPOCHS=150
FOLDS=3
def get_model():
    with strategy.scope():
        rnet = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        # trainable rnet
        rnet.trainable = True
        model = tf.keras.Sequential([
            rnet,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation='softmax',dtype='float32')
        ])
        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(
        optimizer=opt,
        loss = 'sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    return model

def train_cross_validate(folds = 5):
    histories = []
    models = []
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 14)
    kfold = KFold(folds, shuffle = True, random_state = 42)
    for f, (trn_ind, val_ind) in enumerate(kfold.split(TRAINING_FILENAMES)):
        print(); print('#'*25)
        print('### FOLD',f+1)
        print('#'*25)
        train_dataset = load_dataset(list(pd.DataFrame({'TRAINING_FILENAMES': TRAINING_FILENAMES}).loc[trn_ind]['TRAINING_FILENAMES']), labeled = True)
        val_dataset = load_dataset(list(pd.DataFrame({'TRAINING_FILENAMES': TRAINING_FILENAMES}).loc[val_ind]['TRAINING_FILENAMES']), labeled = True, ordered = True)
        checkpoint_name = f'model_fold_{folds + 1}' + '.h5'
        mc = tf.keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
        model = get_model()
        ds = [NUM_TRAINING_IMAGES]
        model.summary()
        train_data = get_training_dataset(ds, train_dataset, augment=True)
        val_data = get_validation_dataset(val_dataset)
        history = model.fit(
            train_data, 
            steps_per_epoch = STEPS_PER_EPOCH,
            epochs = EPOCHS,
            callbacks = [early_stopping, mc],
            validation_data = val_data,
            verbose=2
        )
        model.load_weights(checkpoint_name)
        models.append(model)
        histories.append(history)
    return histories, models

def train_and_predict(folds = 5):
    test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.
    test_images_ds = test_ds.map(lambda image, idnum: image)
    print('Start training %i folds'%folds)
    histories, models = train_cross_validate(folds = folds)
    
    print('Computing predictions...')
    # get the mean probability of the folds models
    probabilities = np.average([models[i].predict(test_images_ds) for i in range(folds)], axis = 0)
    predictions = np.argmax(probabilities, axis=-1)
    print('Generating submission.csv file...')
    test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
    test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
    np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
    return histories, models
    
# run train and predict
if cross_validation:
    histories, models = train_and_predict(folds = FOLDS)
if not cross_validation:
    display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)
    display_training_curves(history.history['sparse_categorical_accuracy'], history.history['val_sparse_categorical_accuracy'], 'accuracy', 212)
# Load the best model
if not cross_validation:
    model.load_weights('best_model.h5')

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
    !head submission.csv
if not cross_validation:
    dataset = get_validation_dataset()
    dataset = dataset.unbatch().batch(20)
    batch = iter(dataset)
if not cross_validation:
    # run this cell again for next set of images
    images, labels = next(batch)
    probabilities = model.predict(images)
    predictions = np.argmax(probabilities, axis=-1)
    display_batch_of_images((images, labels), predictions)
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
%%time
all_labels = []; all_prob = []; all_pred = []
kfold = KFold(FOLDS, shuffle = True, random_state = 42)
for j, (trn_ind, val_ind) in enumerate( kfold.split(TRAINING_FILENAMES) ):
    print('Inferring fold',j+1,'validation images...')
    VAL_FILES = list(pd.DataFrame({'TRAINING_FILENAMES': TRAINING_FILENAMES}).loc[val_ind]['TRAINING_FILENAMES'])
    NUM_VALIDATION_IMAGES = count_data_items(VAL_FILES)
    cmdataset = get_validation_dataset(load_dataset(VAL_FILES, labeled = True, ordered = True))
    images_ds = cmdataset.map(lambda image, label: image)
    labels_ds = cmdataset.map(lambda image, label: label).unbatch()
    all_labels.append( next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() ) # get everything as one batch
    prob = models[j].predict(images_ds)
    all_prob.append( prob )
    all_pred.append( np.argmax(prob, axis=-1) )
cm_correct_labels = np.concatenate(all_labels)
cm_probabilities = np.concatenate(all_prob)
cm_predictions = np.concatenate(all_pred)

print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)
print("Predicted labels: ", cm_predictions.shape, cm_predictions); print()
cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))
score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
display_confusion_matrix(cmat, score, precision, recall)
print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall)); print()
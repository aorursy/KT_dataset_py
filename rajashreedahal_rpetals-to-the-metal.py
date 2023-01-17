# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import math
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature

import re
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    tpu_strategy = tf.distribute.get_strategy() 
    
AUTO = tf.data.experimental.AUTOTUNE
print("REPLICAS: ", tpu_strategy.num_replicas_in_sync)
IMAGE_SIZE=[331,331]
EPOCHS=34
BATCH_SIZE=16*tpu_strategy.num_replicas_in_sync
from kaggle_datasets import KaggleDatasets
GCS_DS_PATH=KaggleDatasets().get_gcs_path('tpu-getting-started')
EXT_GCS = KaggleDatasets().get_gcs_path('tf-flower-photo-tfrec')
GCS_DS_PATH
GCS_PATH_SELECT={
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512',
}
GCS_PATH=GCS_PATH_SELECT[IMAGE_SIZE[0]]
GCS_PATH
training_filenames=tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec' )
validation_filenames=tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
test_filenames=tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')
#imagenet_files = tf.io.gfile.glob(EXT_GCS + '/imagenet/tfrecords-jpeg-224*224/*.tfrec')
#inaturalist_files=tf.io.gfile.glob(EXT_GCS + '/inaturalist/tfrecords-jpeg-224*224/*.tfrec')
#openimage_files=tf.io.gfile.glob(EXT_GCS + '/openimage/tfrecords-jpeg-224*224/*.tfrec')
#tf_flowers_files=tf.io.gfile.glob(EXT_GCS + '/tf_flowers/tfrecords-jpeg-224*224/*.tfrec')
#training_filenames=training_filenames + imagenet_files + inaturalist_files + openimage_files + tf_flowers_files
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
lr_start=0.0022
lr_max=0.0023* tpu_strategy.num_replicas_in_sync
lr_min=0.0022
lr_ramp_up_epoch=9
lr_sustain_epoch=0
lr_exp_decay=.8
def lrfn(epoch):
    if epoch < lr_ramp_up_epoch:
        lr = (lr_max - lr_start) / lr_ramp_up_epoch * epoch + lr_start
    elif epoch < lr_ramp_up_epoch + lr_sustain_epoch:
        lr = lr_max
    else:
        lr = (lr_max-lr_min) * lr_exp_decay ** (epoch - lr_ramp_up_epoch - lr_sustain_epoch) + lr_min
    return lr
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
import matplotlib.pyplot as plt
rng = [i for i in range(34 if EPOCHS<34 else EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
import tensorflow as tf
def decode_image(image_data):
    image=tf.image.decode_jpeg(image_data,channels=3)
    image=tf.cast(image,tf.float32)/255.0 #This converts image to floats in [0,1] range.
    image=tf.reshape(image,[*IMAGE_SIZE,3]) #It is the explicit size needed for TPU
    return image
from tensorflow.io import FixedLenFeature, VarLenFeature
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT={
        'image':tf.io.FixedLenFeature([],tf.string),
        'class':tf.io.FixedLenFeature([],tf.int64),}
    
    example=tf.io.parse_single_example(example,LABELED_TFREC_FORMAT)
    image=decode_image(example['image'])
    label=tf.cast(example['class'],tf.int32)
    return image,label
        
def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT={
        'image':tf.io.FixedLenFeature([],tf.string),
        'id':tf.io.FixedLenFeature([],tf.string),
        
    }
    example=tf.io.parse_single_example(example,UNLABELED_TFREC_FORMAT)
    image=decode_image(example['image'])
    ids=example['id']
    return image,ids
    
def load_dataset(filenames,labeled=True,ordered=False):
    ignore_order=tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic=False
    dataset=tf.data.TFRecordDataset(filenames,num_parallel_reads=AUTO)
    dataset=dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,num_parallel_calls=AUTO)
    return dataset
        
def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, 0, 2)
    return image, label   
def get_training_dataset():
    dataset=load_dataset(training_filenames,labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls = AUTO)
    dataset=dataset.repeat()
    dataset=dataset.shuffle(100)
    dataset=dataset.batch(BATCH_SIZE)
    dataset=dataset.prefetch(AUTO)
    return dataset
def get_validation_dataset():
    dataset=load_dataset(validation_filenames,labeled=True,ordered=False)
    dataset=dataset.batch(BATCH_SIZE)
    dataset=dataset.cache()
    return dataset
def get_test_dataset(ordered=False):
    dataset=load_dataset(test_filenames,labeled=False,ordered=ordered)
    dataset=dataset.batch(BATCH_SIZE)
    dataset=dataset.cache()
    return dataset
train_dataset=get_training_dataset()
validation_dataset=get_validation_dataset()
test_dataset=get_test_dataset()
print(len(CLASSES))


np.set_printoptions(threshold=15,linewidth=80)

def batch_to_numpy_images_and_labels(data):
    images,labels=data
    numpy_images=images.numpy()
    numpy_labels=labels.numpy()
    
    if numpy_labels.dtype==object:
        numpy_labels=[None for i in enumerate(numpy_images)]
    return numpy_images,numpy_labels
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
def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)
NUM_TRAINING_IMAGES = count_data_items(training_filenames)
NUM_VALIDATION_IMAGES = count_data_items(validation_filenames)
NUM_TEST_IMAGES = count_data_items(test_filenames)
STEPS_FOR_EPOCH=NUM_TRAINING_IMAGES// BATCH_SIZE
print('Training_size=',NUM_TRAINING_IMAGES  ,'Validation size=',NUM_VALIDATION_IMAGES , 'Test size=',NUM_TEST_IMAGES )
for image, label in get_training_dataset().take(3):
    print(image.numpy().shape, label.numpy().shape)
print('Training data label examples:', label.numpy())
#

print('Validation data shapes')
for image, label in get_validation_dataset().take(3):
    print(image.numpy().shape, label.numpy().shape)
print('Validation data label examples:', label.numpy())
#

print('Test data shapes')
for image, ids in get_test_dataset().take(3):
    print(image.numpy().shape, ids.numpy().shape)
print('Test data IDs:', ids.numpy().astype('U'))

training_dataset = train_dataset.unbatch().batch(20)
train_batch = iter(training_dataset)
#
display_batch_of_images(next(train_batch))

test_dataset = test_dataset.unbatch().batch(20)
test_batch = iter(test_dataset)
#
display_batch_of_images(next(test_batch))
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights = True)
!pip install efficientnet
import efficientnet.tfkeras
#Building and training the model 
with tpu_strategy.scope():    

    pretrained_model =efficientnet.tfkeras.EfficientNetB7(
        include_top=False, weights='imagenet', input_shape=[*IMAGE_SIZE,3])
    pretrained_model.trainable = False # tramsfer learning
    model=tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(104, activation='softmax')
    ])
            
model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

model.summary()
historical = model.fit(train_dataset, 
          steps_per_epoch=STEPS_FOR_EPOCH, 
          epochs=EPOCHS, 
          validation_data=validation_dataset,callbacks=[lr_callback,early_stop])
with tpu_strategy.scope():    

    pretrained_model =tf.keras.applications.DenseNet201(
        include_top=False, weights='imagenet', input_shape=[*IMAGE_SIZE,3])
    pretrained_model.trainable = False # tramsfer learning
    model2=tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(104, activation='softmax')
    ])
            
model2.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

model2.summary()
historical = model2.fit(train_dataset, 
          steps_per_epoch=STEPS_FOR_EPOCH, 
          epochs=EPOCHS, callbacks=[lr_callback],
          validation_data=validation_dataset)
lr_start1=0.0004
lr_max1=0.001* tpu_strategy.num_replicas_in_sync
lr_min1=0.0006
lr_ramp_up_epoch1=8
lr_sustain_epoch1=0
lr_exp_decay1=.8
def lrfn1(epoch):
    if epoch < lr_ramp_up_epoch1:
        lr = (lr_max1 - lr_start1) / lr_ramp_up_epoch1 * epoch + lr_start1
    elif epoch < lr_ramp_up_epoch1 + lr_sustain_epoch1:
        lr = lr_max1
    else:
        lr = (lr_max1-lr_min1) * lr_exp_decay1 ** (epoch - lr_ramp_up_epoch1 - lr_sustain_epoch1) + lr_min1
    return lr
lr_callback1 = tf.keras.callbacks.LearningRateScheduler(lrfn1, verbose = True)

rng1 = [i for i in range(12 if EPOCHS<12 else EPOCHS)]
y = [lrfn1(x) for x in rng1]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
with tpu_strategy.scope():    

    pretrained_model =tf.keras.applications.ResNet50V2(
        include_top=False, weights='imagenet', input_shape=[*IMAGE_SIZE,3])
    pretrained_model.trainable = False # tramsfer learning
    model3=tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(104, activation='softmax')
    ])
            
model3.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

model3.summary()
historical3 = model3.fit(train_dataset, 
          steps_per_epoch=STEPS_FOR_EPOCH, 
          epochs=EPOCHS,callbacks=[lr_callback1],
          validation_data=validation_dataset)
test_ds = get_test_dataset(ordered=True)
print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
predictions = np.argmax(probabilities, axis=-1)
print(predictions)
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities1 = model.predict(test_images_ds)
predictions1= np.argmax(probabilities1, axis=-1)
print(predictions1)

test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities3 = model3.predict(test_images_ds)
predictions3= np.argmax(probabilities3, axis=-1)
print(predictions3)
probabilities=(probabilities+probabilities3+probabilities1)/3
predictions=np.argmax(probabilities,axis=-1)
print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
#

#print('Generating submission.csv file...')
#test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
#test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
#np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

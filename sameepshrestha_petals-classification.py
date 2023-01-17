# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from kaggle_datasets import KaggleDatasets
import math,re ,os 


!pip install efficientnet
import efficientnet.tfkeras
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

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
GCS_DS_PATH = KaggleDatasets().get_gcs_path()
Image_size=[512,512]

Batch_size=16 * strategy.num_replicas_in_sync

Epochs=20


lr_start=0.0004
lr_max=0.001* strategy.num_replicas_in_sync
lr_min=0.0006
lr_ramp_up_epoch=8
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

rng = [i for i in range(12 if Epochs<12 else Epochs)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))

GCS_DS_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started')

GCS_PATH_SELECT = { 
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'
}

GCS_PATH = GCS_PATH_SELECT[Image_size[0]]

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec') 
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')
CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', 
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose'] 
AUTO = tf.data.experimental.AUTOTUNE
#decoding the encoded image 
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*Image_size, 3]) # explicit size needed for TPU
    return image
#reading a data if it has a label such as class 
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs
#reading a data if it does not have any label 
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
#loading the dataest with a label 

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
def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    return image, label  

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(Batch_size)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=False)
    dataset = dataset.batch(Batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) 
    return dataset
    
def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES , labeled=False, ordered=ordered)
    dataset = dataset.batch(Batch_size)
    dataset = dataset.cache() 
    dataset = dataset.prefetch(AUTO)
    return dataset
def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

train_dataset= get_training_dataset()
validation_dataset=get_validation_dataset()
test_dataset=get_test_dataset()
print(validation_dataset)
print(test_dataset)

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_FOR_EPOCH=NUM_TRAINING_IMAGES// Batch_size
print('Training_size=',NUM_TRAINING_IMAGES  ,'Validation size=',NUM_VALIDATION_IMAGES , 'Test size=',NUM_TEST_IMAGES )
iter(train_dataset.unbatch().batch(10))


#Building and training the model 
with strategy.scope():    

    pretrained_model =efficientnet.tfkeras.EfficientNetB7(
        include_top=False, weights='imagenet', input_shape=[*Image_size,3])
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
historical1 = model.fit(train_dataset, 
          steps_per_epoch=STEPS_FOR_EPOCH, 
          epochs=Epochs, callbacks=[lr_callback],
          validation_data=validation_dataset)
with strategy.scope():    

    pretrained_model =tf.keras.applications.DenseNet201(
        include_top=False, weights='imagenet', input_shape=[*Image_size,3])
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
          epochs=Epochs, callbacks=[lr_callback],
          validation_data=validation_dataset)
with strategy.scope():    

    pretrained_model =tf.keras.applications.ResNet50V2(
        include_top=False, weights='imagenet', input_shape=[*Image_size,3])
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
          epochs=Epochs,
          validation_data=validation_dataset)
from sklearn.metrics import f1_score
cmdataset=get_validation_dataset(ordered=True)
images_ds = cmdataset.map(lambda image, label: image)
labels_ds = cmdataset.map(lambda image, label: label).unbatch()
cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy()
alpha=np.linspace(.4,.7,20)
scores=[]
probab1=model2.predict(images_ds)
probab2=model3.predict(images_ds)
for a in alpha:
    cm_probabilities = a*probab1+(1-a)*probab2
    cm_predictions = np.argmax(cm_probabilities, axis=-1)
    scores.append(f1_score(cm_correct_labels, cm_predictions, labels=range((104)), average='macro'))
print(scores)
best_alpha = np.argmax(scores)/100
plt.plot(alpha,scores)
best_alpha
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model2.predict(test_images_ds)
predictions2 = np.argmax(probabilities, axis=-1)
print(predictions2)

#print('Generating submission.csv file...')
#test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
#test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
#np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities1 = model.predict(test_images_ds)
predictions1= np.argmax(probabilities1, axis=-1)
print(predictions1)

#print('Generating submission.csv file...')
#test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
#test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
#np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities3 = model3.predict(test_images_ds)
predictions3= np.argmax(probabilities3, axis=-1)
print(predictions3)

#print('Generating submission.csv file...')
#test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
#test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
#np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
probabilities=.525*probabilities+(1-.525)*probabilities3
predictions=np.argmax(probabilities,axis=-1)
print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

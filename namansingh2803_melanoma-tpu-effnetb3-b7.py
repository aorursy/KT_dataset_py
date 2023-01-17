!pip install -q efficientnet
import os

import re

import seaborn as sns

import numpy as np

import pandas as pd

import math



from matplotlib import pyplot as plt



from sklearn import metrics

from sklearn.model_selection import train_test_split



import tensorflow as tf

import tensorflow.keras.layers as L

import tensorflow.keras.backend as K 

from tensorflow.keras.applications import DenseNet121, DenseNet201



import efficientnet.tfkeras as efn



from kaggle_datasets import KaggleDatasets
print(os.listdir('/kaggle/input/siim-isic-melanoma-classification/tfrecords'))
print(os.listdir('/kaggle/input/512x512-melanoma-tfrecords-70k-images/'))
print(os.listdir('/kaggle/input/melanoma-768x768/'))
try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
# For tf.dataset

AUTO = tf.data.experimental.AUTOTUNE



# Data access

#GCS_PATH = KaggleDatasets().get_gcs_path('512x512-melanoma-tfrecords-70k-images')

GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-768x768')



# Configuration

EPOCHS = 10

BATCH_SIZE = 8 * strategy.num_replicas_in_sync

#IMAGE_SIZE = [512, 512]

IMAGE_SIZE = [768,768]
HEIGHT = IMAGE_SIZE[0]

WIDTH = IMAGE_SIZE[1]

CHANNELS = 3
def append_path(pre):

    return np.vectorize(lambda file: os.path.join(GCS_DS_PATH, pre, file))
sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
array = sub.image_name

array.tolist()

print(array)
train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
sns.countplot(train['target'])
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')



CLASSES = [0,1]   
TRAINING_FILENAMES
TRAINING_FILENAMES
def transform_rotation(image):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated

    DIM = IMAGE_SIZE[0]

    XDIM = DIM%2 #fix for size 331

    

    rotation = 15. * tf.random.normal([1],dtype='float32')

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    

    # ROTATION MATRIX

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(rotation_matrix,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES 

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3])



def transform_shear(image):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly sheared

    DIM = IMAGE_SIZE[0]

    XDIM = DIM%2 #fix for size 331

    

    shear = 5. * tf.random.normal([1],dtype='float32')

    shear = math.pi * shear / 180.

        

    # SHEAR MATRIX

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(shear_matrix,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES 

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3])



def transform_shift(image):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly shifted

    DIM = IMAGE_SIZE[0]

    XDIM = DIM%2 #fix for size 331

    

    height_shift = 16. * tf.random.normal([1],dtype='float32') 

    width_shift = 16. * tf.random.normal([1],dtype='float32') 

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

        

    # SHIFT MATRIX

    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(shift_matrix,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES 

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3])



def transform_zoom(image):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly zoomed

    DIM = IMAGE_SIZE[0]

    XDIM = DIM%2 #fix for size 331

    

    height_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    width_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

        

    # ZOOM MATRIX

    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(zoom_matrix,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES 

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3])


def data_augment_chaotic(image, label):

    #image = decode_image_file(filename)

    p_spatial = tf.random.uniform([1], minval=0, maxval=1, dtype='float32')

    p_spatial2 = tf.random.uniform([1], minval=0, maxval=1, dtype='float32')

    p_pixel = tf.random.uniform([1], minval=0, maxval=1, dtype='float32')

    p_crop = tf.random.uniform([1], minval=0, maxval=1, dtype='float32')

    

    ### Spatial-level transforms

    if p_spatial >= .2:

        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_flip_up_down(image)

        

    if p_crop >= .7:

        if p_crop >= .95:

            image = tf.image.random_crop(image, size=[int(HEIGHT*.6), int(WIDTH*.6), CHANNELS])

        elif p_crop >= .85:

            image = tf.image.random_crop(image, size=[int(HEIGHT*.7), int(WIDTH*.7), CHANNELS])

        elif p_crop >= .8:

            image = tf.image.random_crop(image, size=[int(HEIGHT*.8), int(WIDTH*.8), CHANNELS])

        else:

            image = tf.image.random_crop(image, size=[int(HEIGHT*.9), int(WIDTH*.9), CHANNELS])

        image = tf.image.resize(image, size=[HEIGHT, WIDTH])



    if p_spatial2 >= .6:

        if p_spatial2 >= .9:

            image = transform_rotation(image)

        elif p_spatial2 >= .8:

            image = transform_zoom(image)

        elif p_spatial2 >= .7:

            image = transform_shift(image)

        else:

            image = transform_shear(image)

        

    ## Pixel-level transforms

    if p_pixel >= .4:

        if p_pixel >= .85:

            image = tf.image.random_saturation(image, lower=0, upper=2)

        elif p_pixel >= .65:

            image = tf.image.random_contrast(image, lower=.8, upper=2)

        elif p_pixel >= .5:

            image = tf.image.random_brightness(image, max_delta=.2)

        else:

            image = tf.image.adjust_gamma(image, gamma=.6)



    return image, label
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def decode_image_file(file,label=None):

    bits = tf.io.read_file(file)

    image = tf.image.decode_jpeg(bits,channels=3)

    image = tf.cast(image,tf.float32)/255.0

    image = tf.reshape(image,[*IMAGE_SIZE,3])

    if label is None:

        return image

    return image,label



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        #"class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    #label = tf.cast(example['class'], tf.int32)

    label = tf.cast(example['target'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['image_name']

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



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    #image = tf.image.random_flip_up_down(image)

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



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES))
def build_lrfn(lr_start=0.00001, lr_max=0.0001, 

               lr_min=0.000001, lr_rampup_epochs=20, 

               lr_sustain_epochs=0, lr_exp_decay=.8):

    lr_max = lr_max * strategy.num_replicas_in_sync



    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    

    return lrfn
with strategy.scope():

    model = tf.keras.Sequential([

        efn.EfficientNetB3(

            input_shape=(*IMAGE_SIZE, 3),

            #weights='imagenet',

            weights='imagenet',

            include_top=False

        ),

        L.GlobalAveragePooling2D(),

        #L.Dense(2048, activation='relu'),

        #L.Dense(1024, activation='relu'),

        L.Dense(512, activation='relu'),

        #L.Dense(256, activation='relu'),

        L.Dense(128,activation='relu'),

        #L.Dropout(rate=0.5),

        L.Dense(1, activation='sigmoid')

    ])

    

model.compile(

    optimizer='adam',

    loss = 'binary_crossentropy',

    metrics=['accuracy']

)

model.summary()
with strategy.scope():

    model2 = tf.keras.Sequential([

        efn.EfficientNetB3(

            input_shape=(*IMAGE_SIZE, 3),

            #weights='imagenet',

            weights='imagenet',

            include_top=False

        ),

        L.GlobalAveragePooling2D(),

        L.Dense(512, activation='relu'),

        L.Dropout(0.3),

        L.Dense(256, activation='relu'),

        L.Dropout(0.25),

        L.Dense(128, activation='relu'),

        L.Dropout(0.2),

        L.Dense(1, activation='sigmoid')

    ])

    

model2.compile(

    optimizer='adam',

    loss = 'binary_crossentropy',

    metrics=['accuracy']

)

model2.summary()
lrfn = build_lrfn()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
history = model.fit(

    get_training_dataset(), 

    epochs=EPOCHS, 

    callbacks=[lr_schedule],

    steps_per_epoch=STEPS_PER_EPOCH

    #validation_data=valid_dataset

) 
''''history2 = model2.fit(

    get_training_dataset(), 

    epochs=EPOCHS, 

    callbacks=[lr_schedule],

    steps_per_epoch=STEPS_PER_EPOCH

    #validation_data=valid_dataset

) '''
test_ds = get_test_dataset(ordered=True)



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

#probabilities2 = model2.predict(test_images_ds)
sub1 = sub.copy()

sub2 = sub.copy()
print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})

pred_df.head()
#pred_df2 = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities2)})

#pred_df2.head()
sub1.head()
sub2.head()
sub4 = pd.read_csv('/kaggle/input/submissionb7/submissionB7-2.csv')
sub5 = pd.read_csv('/kaggle/input/submissionb3/submissionB3.csv')
sub5
del sub1['target']

sub1 = sub1.merge(pred_df, on='image_name')



sub1.head()
sub1.reindex(array)

sub1.head()
sub1.to_csv('submission-B5.csv',index=False)
#del sub2['target']

#sub2 = sub2.merge(pred_df2, on='image_name')

#sub2.to_csv('submission_EffnetB0.csv',index=False)

#sub2.head()
sub3 = pd.read_csv('/kaggle/input/subeffnetb0/submissionB0.csv')
sub3
#sub_es = sub1[['image_name']]

#sub_es['target'] = 0.55*sub4['target'] + 0.45*sub5['target'] #sub1 is DenseNet201 sub3 is B0 and sub4 is B7 sub2 is B0 trained again sub5 is DenseNet 201 trained

#sub_es.to_csv('submission.csv', index=False)
model.save('EffNetB5-Melanoma.h5')

#model2.save('EffnetB0-Melanoma.h5')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import tensorflow as tf, tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D

from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.utils import plot_model

from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from kaggle_datasets import KaggleDatasets

import matplotlib.pyplot as plt

import random

import math

AUTO = tf.data.experimental.AUTOTUNE
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



print("REPLICAS: ", strategy.num_replicas_in_sync)
IMAGE_SIZE = 331

EPOCHS = 30

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



NUM_TRAINING_IMAGES = 12753

NUM_TEST_IMAGES = 7382

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
GCS_DS_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started')

print(GCS_DS_PATH)
train_data = tf.data.TFRecordDataset(

    tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-' + str(IMAGE_SIZE) + 'x' + str(IMAGE_SIZE) + '/train/*.tfrec'),

    num_parallel_reads = tf.data.experimental.AUTOTUNE

)
# disable order and increase speed

ignore_order = tf.data.Options()

ignore_order.experimental_deterministic = False 

train_data = train_data.with_options(ignore_order)
def read_labeled_tfrecord(example):

    tfrec_format = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "class": tf.io.FixedLenFeature([], tf.int64), 

    }

    

    example = tf.io.parse_single_example(example, tfrec_format)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    

    # returns a dataset of (image, label) pairs

    return image, label 





def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  

    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])

    

    return image
train_data = train_data.map(read_labeled_tfrecord)
def random_blockout(img, sl=0.1, sh=0.2, rl=0.4):

    p=random.random()

    if p>=0.25:

        h, w, c = IMAGE_SIZE, IMAGE_SIZE, 3

        origin_area = tf.cast(h*w, tf.float32)



        e_size_l = tf.cast(tf.round(tf.sqrt(origin_area * sl * rl)), tf.int32)

        e_size_h = tf.cast(tf.round(tf.sqrt(origin_area * sh / rl)), tf.int32)



        e_height_h = tf.minimum(e_size_h, h)

        e_width_h = tf.minimum(e_size_h, w)



        erase_height = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_height_h, dtype=tf.int32)

        erase_width = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_width_h, dtype=tf.int32)



        erase_area = tf.zeros(shape=[erase_height, erase_width, c])

        erase_area = tf.cast(erase_area, tf.uint8)



        pad_h = h - erase_height

        pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)

        pad_bottom = pad_h - pad_top



        pad_w = w - erase_width

        pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)

        pad_right = pad_w - pad_left



        erase_mask = tf.pad([erase_area], [[0,0],[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1)

        erase_mask = tf.squeeze(erase_mask, axis=0)

        erased_img = tf.multiply(tf.cast(img,tf.float32), tf.cast(erase_mask, tf.float32))



        return tf.cast(erased_img, img.dtype)

    else:

        return tf.cast(img, img.dtype)
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transformmatrix which transforms indicies

        

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    shear = math.pi * shear / 180.

    

    # ROTATION MATRIX

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )

        

    # SHEAR MATRIX

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    

    

    # ZOOM MATRIX

    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )

    

    # SHIFT MATRIX

    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )

    

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))
def augment(image, label):

    image = tf.image.random_flip_left_right(image)

    

    #image = tf.image.random_brightness(image, max_delta=0.5)

    #image = tf.image.random_saturation(image, lower=0.2, upper=0.5)

    

    #image= random_blockout(image)

    

    DIM = IMAGE_SIZE

    XDIM = DIM%2 #fix for size 331

    

    rot = 15. * tf.random.normal([1],dtype='float32')

    shr = 5. * tf.random.normal([1],dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    h_shift = 16. * tf.random.normal([1],dtype='float32') 

    w_shift = 16. * tf.random.normal([1],dtype='float32') 

  

    # GET TRANSFORMATION MATRIX

    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES           

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image,tf.transpose(idx3))

    #image = tf.image.resize_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return tf.reshape(d,[DIM,DIM,3]), label
train_data = train_data.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_data = train_data.repeat()

train_data = train_data.shuffle(2048)

train_data = train_data.batch(BATCH_SIZE)

train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
fig, axes = plt.subplots(1, 5, figsize=(15, 5))



for images, labels in train_data.take(1):

    for i in range(5):

        axes[i].set_title('Label: {0}'.format(labels[i]))

        axes[i].imshow(images[i])
val_data = tf.data.TFRecordDataset(

    tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-' + str(IMAGE_SIZE) + 'x' + str(IMAGE_SIZE) + '/val/*.tfrec'),

    num_parallel_reads = tf.data.experimental.AUTOTUNE

)

val_data = val_data.with_options(ignore_order)

val_data = val_data.map(read_labeled_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)

val_data = val_data.batch(BATCH_SIZE)

val_data = val_data.cache()

val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)
with strategy.scope():

    model = tf.keras.applications.InceptionV3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')

    for layer in model.layers:

        layer.trainable=False

    last_layer=model.get_layer('mixed7')

    last_output = last_layer.output

    x=layers.Flatten()(last_output)

    x=layers.Dense(1024, activation='relu')(x)

    x=layers.Dropout(0.2)(x)

    x=layers.Dense(104, activation='softmax')(x)

    model=Model(model.input, x)

    opt = Adam(lr=0.0005)

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()
callbacks=[ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1), 

           EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)]
history=model.fit(train_data, validation_data=val_data, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=callbacks)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))



axes[0].plot(history.history['loss'], label='train')

axes[0].plot(history.history['val_loss'], label='val')

axes[0].set_title('loss')

axes[0].legend()

axes[1].plot(history.history['accuracy'], label='train')

axes[1].plot(history.history['val_accuracy'], label='val')

axes[1].set_title('accuracy')

axes[1].legend()
def read_unlabeled_tfrecord(example):

    tfrec_format = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "id": tf.io.FixedLenFeature([], tf.string),  

    }

    

    example = tf.io.parse_single_example(example, tfrec_format)

    image = decode_image(example['image'])

    idnum = example['id']

    

    return image, idnum
test_data = tf.data.TFRecordDataset(

    tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-' + str(IMAGE_SIZE) + 'x' + str(IMAGE_SIZE) + '/test/*.tfrec'),

    num_parallel_reads = tf.data.experimental.AUTOTUNE

)



test_data = test_data.with_options(tf.data.Options())

test_data = test_data.map(read_unlabeled_tfrecord, num_parallel_calls = tf.data.experimental.AUTOTUNE)

test_data = test_data.batch(BATCH_SIZE)

test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

test_images = test_data.map(lambda image, idnum: image)
test_images
CLASSES = [

    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 

    'wild geranium', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 

    'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 

    'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 

    'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 

    'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 

    'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 

    'carnation', 'garden phlox', 'love in the mist', 'cosmos',  'alpine sea holly', 

    'ruby-lipped cattleya', 'cape flower', 'great masterwort',  'siam tulip', 

    'lenten rose', 'barberton daisy', 'daffodil',  'sword lily', 'poinsettia', 

    'bolero deep blue',  'wallflower', 'marigold', 'buttercup', 'daisy', 

    'common dandelion', 'petunia', 'wild pansy', 'primula',  'sunflower', 

    'lilac hibiscus', 'bishop of llandaff', 'gaura',  'geranium', 'orange dahlia', 

    'pink-yellow dahlia', 'cautleya spicata',  'japanese anemone', 'black-eyed susan', 

    'silverbush', 'californian poppy',  'osteospermum', 'spring crocus', 'iris', 

    'windflower',  'tree poppy', 'gazania', 'azalea', 'water lily',  'rose', 

    'thorn apple', 'morning glory', 'passion flower',  'lotus', 'toad lily', 

    'anthurium', 'frangipani',  'clematis', 'hibiscus', 'columbine', 'desert-rose', 

    'tree mallow', 'magnolia', 'cyclamen ', 'watercress',  'canna lily', 

    'hippeastrum ', 'bee balm', 'pink quill',  'foxglove', 'bougainvillea', 

    'camellia', 'mallow',  'mexican petunia',  'bromelia', 'blanket flower', 

    'trumpet creeper',  'blackberry lily', 'common tulip', 'wild rose']
pred=model.predict(test_images)
np.shape(pred)
fig, axes = plt.subplots(1, 6, figsize=(15, 6))



for images in test_images.take(1):

    for i in range(6):

        axes[i].set_title(CLASSES[np.argmax(pred[i])])

        axes[i].imshow(images[i])

predictions=np.argmax(pred, axis=-1)
np.shape(predictions)
ids = []



for image, image_ids in test_data.take(NUM_TEST_IMAGES):

    ids.append(image_ids.numpy())



ids = np.concatenate(ids, axis=None).astype(str)
submission = pd.DataFrame(data={'id': ids, 'label': predictions})

submission.to_csv('submission.csv', index=False)
model.save('flower_model.h5')
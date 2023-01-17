!pip install -q efficientnet
import math, re, os



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

import tensorflow as tf

import tensorflow.keras.layers as L

import efficientnet.tfkeras as efn

from sklearn import metrics

from sklearn.model_selection import train_test_split
AUTO = tf.data.experimental.AUTOTUNE



# Create strategy from tpu

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)

strategy = tf.distribute.experimental.TPUStrategy(tpu)



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()



# Configuration

EPOCHS = 80

BATCH_SIZE = 16 * strategy.num_replicas_in_sync
def format_path(st):

    return GCS_DS_PATH + '/images/' + st + '.jpg'
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')



train_paths = train.image_id.apply(format_path).values

test_paths = test.image_id.apply(format_path).values



train_labels = train.loc[:, 'healthy':].values
def decode_image(filename, label=None, image_size=(512, 512)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.rgb_to_yuv(image)

    image = tf.image.resize(image, image_size)

    

    if label is None:

        return image

    else:

        return image, label



def data_augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    

    if label is None:

        return image

    else:

        return image, label
test_dataset = (

        tf.data.Dataset

        .from_tensor_slices(test_paths)

        .map(decode_image, num_parallel_calls=AUTO)

        .map(data_augment, num_parallel_calls=AUTO)

        .batch(BATCH_SIZE)

)
def build_lrfn(lr_start=0.00001, lr_max=0.00005, 

               lr_min=0.000001, lr_rampup_epochs=3, 

               lr_sustain_epochs=2, lr_exp_decay=.86):

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
from tensorflow.keras.callbacks import Callback 



class TTACallback(Callback):

    def __init__(self, test_data, score_thr):

        self.test_data = test_data

        self.score_thr = score_thr

        self.test_pred = []

        

    def on_epoch_end(self, epoch, logs=None):

        if logs['val_categorical_accuracy'] > self.score_thr:

            print('Run TTA...')

            self.test_pred.append(self.model.predict(self.test_data))
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Layer



class SoftProbField(Layer):

    # https://www.kaggle.com/miklgr500/plant-pathology-very-concise-tpu-efficientnetl2

    def __init__(self, **kwargs):

        super(SoftProbField, self).__init__(**kwargs)



    def build(self, input_shape):

        super(SoftProbField, self).build(input_shape)  # Be sure to call this at the end



    def call(self, x):

        h = x[:, 0]

        s = x[:, 1]

        r = x[:, 2]

        

        m = s*r*(1-h)

        s = s*(1-h)*(1-m)

        r = r*(1-h)*(1-m)

        return tf.stack([h, s, r, m], axis=-1)



    def compute_output_shape(self, input_shape):

        return (input_shape[0], 4)
with strategy.scope():

    model = tf.keras.Sequential([

        efn.EfficientNetB5(

            input_shape=(512, 512, 3),

            weights='noisy-student',

            include_top=False

        ),

        L.GlobalMaxPooling2D(),

        L.Dense(3**3),

        L.LayerNormalization(),

        L.LeakyReLU(0.1),

        L.Dropout(0.5),

        L.Dense(3, activation='sigmoid'),

        SoftProbField()

    ])

        

    model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

    model.summary()
import matplotlib.pyplot as plt



plt.figure(figsize=(10, 7))



lrfn = build_lrfn()

plt.plot([i for i in range(30)], [lrfn(i) for i in range(30)]);
weights = model.get_weights()
IMAGE_SIZE = [512, 512]



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



def transform(image,label):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, sheared, zoomed, and shifted

    DIM = IMAGE_SIZE[0]

    XDIM = DIM%2 #fix for size 331

    

    rot = 45. * tf.random.normal([1],dtype='float32')

    shr = 32. * tf.random.normal([1],dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/5.

    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/5.

    h_shift = 32. * tf.random.normal([1],dtype='float32') 

    w_shift = 32. * tf.random.normal([1],dtype='float32') 

  

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

        

    return tf.reshape(d,[DIM,DIM,3]),label
def get_datasets(tr_idx, vl_idx):

    train_dataset = (

        tf.data.Dataset

        .from_tensor_slices((train_paths[tr_idx], train_labels[tr_idx]))

        .map(decode_image, num_parallel_calls=AUTO)

        .cache()

        .map(data_augment, num_parallel_calls=AUTO)

        .map(transform, num_parallel_calls=AUTO)

        .repeat()

        .shuffle(512)

        .batch(64)

        .prefetch(AUTO)

    )



    valid_dataset = (

        tf.data.Dataset

        .from_tensor_slices((train_paths[vl_idx], train_labels[vl_idx]))

        .map(decode_image, num_parallel_calls=AUTO)

        .batch(64)

        .cache()

        .prefetch(AUTO)

    )

    return train_dataset, valid_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold



gkf = StratifiedKFold(n_splits=5)

tta = TTACallback(test_dataset, 0.96)

STEPS_PER_EPOCH = train_labels.shape[0] // 64



for i, (tr_idx, vl_idx) in enumerate(gkf.split(train_paths, train_labels.argmax(-1))):

    print(f'Start {i} fold')

    tf.keras.backend.clear_session()

    model.set_weights(weights)

    train_dataset, valid_dataset = get_datasets(tr_idx, vl_idx)

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

    history = model.fit(

                    train_dataset, 

                    epochs=20, 

                    callbacks=[lr_schedule, tta],

                    steps_per_epoch=STEPS_PER_EPOCH * 6,

                    validation_data=valid_dataset

    )
probs = np.mean(tta.test_pred, axis=0)

sub.loc[:, 'healthy':] = probs

sub.to_csv('submission.csv', index=False)

sub.head()
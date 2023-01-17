!pip install efficientnet > /dev/null

import os
import random
import re
import math
import time

import numpy as np
import pandas as pd
import PIL as pil
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as be
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.callbacks import *

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from kaggle_datasets import KaggleDatasets
from tqdm import tqdm

import efficientnet.tfkeras as efn

SEED = 42
random.seed(a=SEED)
local_path = '/kaggle/input/siim-isic-melanoma-classification'
df_train = pd.read_csv(os.path.join(local_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(local_path, 'test.csv'))
df_submission = pd.read_csv(os.path.join(local_path, 'sample_submission.csv'))

processed_path = KaggleDatasets().get_gcs_path('melanoma-256x256')
training = np.sort(np.array(tf.io.gfile.glob(processed_path + '/train*.tfrec')))
testing = np.sort(np.array(tf.io.gfile.glob(processed_path + '/test*.tfrec')))
FOLDS = 5

X_train = [None] * FOLDS
X_val = [None] * FOLDS

kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (id_train, id_val) in enumerate(kfold.split(np.arange(len(training)))):
    X_train[fold] = tf.io.gfile.glob([processed_path + '/train%.2i*.tfrec' % x for x in id_train])
    X_val[fold] = tf.io.gfile.glob([processed_path + '/train%.2i*.tfrec' % x for x in id_val])
    
X_test = testing
class Processor():
    def __init__(self, device='TPU'):
        self.device = device
        self.tpu_resolve()
        self.tpu_initialize()
        self.gpu()
        self.replicas = 1
        self.strategy = []

    def tpu_resolve(self):
        try:
            self.tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU', self.tpu.master())
        except ValueError:
            self.tpu = None
            self.device = 'GPU'
        
    def tpu_initialize(self):
        if self.tpu and self.device == 'TPU':
            try:
                tf.config.experimental_connect_to_cluster(self.tpu)
                tf.tpu.experimental.initialize_tpu_system(self.tpu)
                self.strategy = tf.distribute.experimental.TPUStrategy(self.tpu)
                self.replicas = self.strategy.num_replicas_in_sync
            except _:
                print('Failed to initialize TPU Cluster')
                self.device = 'GPU'
                
    def gpu(self):
        if self.device == 'GPU':
            gpu_devices = len(tf.config.experimental.list_physical_devices('GPU'))
            if gpu_devices > 0:
                self.strategy = tf.distribute.get_strategy()
                self.replicas = self.strategy.num_replicas_in_sync
                print('Connected to GPU with', gpu_devices, 'devices')
            else:
                self.device = 'CPU'
                print('Connected to CPU')
        
proc = Processor()
class Dataset():
    def __init__(self, files, is_training=True, augment=False, batch_size=16):
        self.is_training = is_training
        self.files = files
        
        auto = tf.data.experimental.AUTOTUNE

        ds = tf.data.TFRecordDataset(files, num_parallel_reads=auto)
        ds = ds.cache()
        ds = ds.map(self.read, num_parallel_calls=auto)
        ds = ds.map( lambda img, img_or_target: (self.prepare_image(img, augment), img_or_target) , 
                    num_parallel_calls=auto)
        ds = ds.batch(16)
        ds = ds.prefetch(auto)
        self.dataset = ds
    
    def stream(self):
        return self.dataset
        
    def read(self, item):
        if self.is_training:
            item = tf.io.parse_single_example(item, {
                'image': tf.io.FixedLenFeature([], tf.string),
                'target': tf.io.FixedLenFeature([], tf.int64),
            })
            return item['image'], item['target']
        else:
            item = tf.io.parse_single_example(item, {
                'image': tf.io.FixedLenFeature([], tf.string),
                'image_name': tf.io.FixedLenFeature([], tf.string),        
            })
            return item['image'], item['image_name']
    
    def prepare_image(self, img, augment=False):
        read = 256

        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        
        if augment:
            img = self.transform(img, read)
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_saturation(img, 0.7, 1.3)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_brightness(img, 0.1)
    
        img = tf.reshape(img, [read, read, 3])
        return img

    def count(self):
        n = [ int(re.compile(r"-([0-9]*)\.").search(name).group(1)) for name in self.files ]
        return np.sum(n)
    
    def transform(self, image, DIM=256):
        # https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords
        # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
        # output - image randomly rotated, sheared, zoomed, and shifted
        XDIM = DIM%2 #fix for size 331

        rot = 180.0 * tf.random.normal([1], dtype='float32')
        shr = 2.0 * tf.random.normal([1], dtype='float32') 
        h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 8.0
        w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 8.0
        h_shift = 8.0 * tf.random.normal([1], dtype='float32') 
        w_shift = 8.0 * tf.random.normal([1], dtype='float32') 

        # GET TRANSFORMATION MATRIX
        m = self.get_matrix(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

        # LIST DESTINATION PIXEL INDICES
        x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
        y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
        z   = tf.ones([DIM*DIM], dtype='int32')
        idx = tf.stack( [x,y,z] )

        # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
        idx2 = be.dot(m, tf.cast(idx, dtype='float32'))
        idx2 = be.cast(idx2, dtype='int32')
        idx2 = be.clip(idx2, -DIM//2+XDIM+1, DIM//2)

        # FIND ORIGIN PIXEL VALUES           
        idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
        d    = tf.gather_nd(image, tf.transpose(idx3))

        return tf.reshape(d,[DIM, DIM,3])
    
    
    def get_matrix(self, rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
        # https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords
        # returns 3x3 transformmatrix which transforms indicies
        
        # CONVERT DEGREES TO RADIANS
        rotation = math.pi * rotation / 180.
        shear    = math.pi * shear    / 180.

        def get_3x3_mat(lst):
            return tf.reshape(tf.concat([lst],axis=0), [3,3])
    
        # ROTATION MATRIX
        c1   = tf.math.cos(rotation)
        s1   = tf.math.sin(rotation)
        one  = tf.constant([1],dtype='float32')
        zero = tf.constant([0],dtype='float32')

        rotation_matrix = get_3x3_mat([c1,   s1,   zero, 
                                       -s1,  c1,   zero, 
                                       zero, zero, one])    
        # SHEAR MATRIX
        c2 = tf.math.cos(shear)
        s2 = tf.math.sin(shear)    

        shear_matrix = get_3x3_mat([one,  s2,   zero, 
                                    zero, c2,   zero, 
                                    zero, zero, one])        
        # ZOOM MATRIX
        zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 
                                   zero,            one/width_zoom, zero, 
                                   zero,            zero,           one])    
        # SHIFT MATRIX
        shift_matrix = get_3x3_mat([one,  zero, height_shift, 
                                    zero, one,  width_shift, 
                                    zero, zero, one])

        return be.dot(be.dot(rotation_matrix, shear_matrix), 
                     be.dot(zoom_matrix,     shift_matrix))


ds = Dataset(training, augment=True).stream()
ds = ds.unbatch()
ds = ds.take(1)

for idx, data in enumerate(iter(ds)):
    img, img_or_target = data
    img = np.clip(img.numpy() * 255, 0, 255).astype(np.uint8)
    img = pil.Image.fromarray(img)
    img = img.resize((400,400), resample=pil.Image.BILINEAR)
    plt.imshow(img)
plt.show()
ds = Dataset(testing, is_training=False).stream()
ds = ds.unbatch()
ds = ds.take(1)

for idx, data in enumerate(iter(ds)):
    img, img_or_target = data
    img = np.clip(img.numpy() * 255, 0, 255).astype(np.uint8)
    img = pil.Image.fromarray(img)
    img = img.resize((400,400), resample=pil.Image.BILINEAR)
    plt.imshow(img)
plt.show()
print(
    Dataset(training).count(),
    Dataset(testing, is_training=False).count()
)
df_train['target'].value_counts()
class Model():
    def __init__(self, name):
        self.name = name
        i = Input(shape=(400, 400, 3))
        base = efn.EfficientNetB2(weights='imagenet', input_shape=(400, 400, 3), include_top=False)
        x = base(i)
        x = GlobalAveragePooling2D()(x)
#         x = BatchNormalization()(x)
#         x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
#         x = BatchNormalization()(x)
#         x = Dropout(0.1)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=i, outputs=x)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=BinaryCrossentropy(label_smoothing=0.05),
            metrics=['binary_crossentropy', AUC(name='auc') ] # 'accuracy'
        )
        self.model = model

    def get(self):
        return self.model
        
    def fit(self, train, validate, epochs=10, batch_size=64):
        self.batch_size = batch_size
        # tf.keras.utils.plot_model(self.model, show_shapes=True)
        history = self.model.fit(
            train.stream(),
            validation_data=validate.stream(),
            verbose=1,
            epochs=epochs,
            steps_per_epoch=train.count()/batch_size,
            batch_size=batch_size,
            callbacks=[
#                 EarlyStopping(monitor='auc', mode='max', patience=6, verbose=2, restore_best_weights=True),
                ModelCheckpoint(monitor='val_auc', verbose=1, save_best_only=True, mode='max', 
                                filepath='{val_auc:.5f}_'+self.name+'.h5'),
                # ReduceLROnPlateau(monitor='auc', factor=0.5, patience=3, verbose=1, mode='auto', cooldown=1, min_lr=0 ),
                self.get_learning_rate(),
            ],
        )
        return history
    
    def get_learning_rate(self):
        lr_start = 0.000005
        lr_max = 0.00000125 * self.batch_size
        lr_min = 0.000001
        lr_ramp_ep = 5
        lr_sus_ep = 0
        lr_decay = 0.8
   
        def lrfn(epoch):
            if epoch < lr_ramp_ep:
                lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            elif epoch < lr_ramp_ep + lr_sus_ep:
                lr = lr_max
            else:
                lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            return lr

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
        return lr_callback
# !rm 0.4*.h5
# !rm 0.5*.h5
# !rm 0.6*.h5
# !rm 0.7*.h5
# !rm 0.8*.h5
# # !rm *.png
nn = Model('efn_pool_batch_256BD10x1')

BATCH_SIZE=256

for i in range(FOLDS):
    train = Dataset(X_train[i], augment=True, batch_size=BATCH_SIZE)
    validate = Dataset(X_val[i], batch_size=BATCH_SIZE)
    history = nn.fit(train, validate, epochs=30, batch_size=BATCH_SIZE)
    
    print(history.history)
    
    plt.plot(range(len(history.history['val_auc'])), history.history['val_auc'], color='green')
    plt.plot(range(len(history.history['auc'])), history.history['auc'], color='red')
    plt.plot(range(len(history.history['binary_crossentropy'])), history.history['binary_crossentropy'], color='blue')
    plt.plot(range(len(history.history['val_binary_crossentropy'])), history.history['val_binary_crossentropy'], color='purple')
    plt.show()
# mm = nn.get()
# sub_model = tf.keras.models.load_model('0.90119_efn_pool_batch_256BD10x1.h5') # 0.8681
# sub_model = tf.keras.models.load_model('0.89918_efn_pool_batch_256BD10x1.h5') # 0.8591
# Longer epochs, 16
# sub_model = tf.keras.models.load_model('0.92810_efn_pool_batch_256BD10x1.h5') # 0.9055
sub_model = tf.keras.models.load_model('0.93747_efn_pool_batch_256BD10x1.h5') # 0.9074
ds = Dataset(testing, is_training=False)
predicted = sub_model.predict(ds.stream(), verbose=1)
sub = pd.DataFrame(dict(
    image_name = np.array([img_name.numpy().decode('utf-8') for img, img_name in iter(ds.stream().unbatch()) ]),
    target = np.reshape(predicted, (1, predicted.shape[0]))[0]
))

sub.to_csv('efn_pool_batch3_submission.csv', index=False)
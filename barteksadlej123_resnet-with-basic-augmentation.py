import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as display
import scipy.ndimage as ndimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from kaggle_datasets import KaggleDatasets
from tqdm import tqdm
import math
import keras.backend as K
import re
from datetime import datetime
import time
import kerastuner as kt
for i in os.listdir('../input/siim-isic-melanoma-classification'):
    print(f'../input/siim-isic-melanoma-classification/{i}')

train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
sample_submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')


tfrecords_path = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
files_train = np.sort(np.array(tf.io.gfile.glob(tfrecords_path + '/tfrecords/train*.tfrec')))
files_test  = np.sort(np.array(tf.io.gfile.glob(tfrecords_path + '/tfrecords/test*.tfrec')))
print(tfrecords_path)

DEVICE = "TPU"

CFG = dict(
    batch_size        =  128,
    
    read_size         = 1024, 
    crop_size         = 700, 
    net_size          = 512, 
    
    epochs            =  30,
    LR_START          =   0.000050,
    LR_MAX            =   0.000200,
    LR_MIN            =   0.000010,
    LR_RAMPUP_EPOCHS  =   5,
    LR_SUSTAIN_EPOCHS =   0,
    LR_EXP_DECAY      =   0.8,
    
    rot               = 180.0,
    shr               =   2.0,
    hzoom             =   8.0,
    wzoom             =   8.0,
    hshift            =   8.0,
    wshift            =   8.0,

    optimizer         = 'adam',
    label_smooth_fac  =   0.05,
    
    tta_steps         =  25,
    
    validation_split=0.2
)
train.head()
sns.countplot(x='benign_malignant',data = train)
sns.countplot(x='sex',data = train)
sns.distplot(train["age_approx"])
benign_df = train[train.benign_malignant == 'benign']
malignant_df = train[train.benign_malignant == 'malignant']
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

plt.subplot(1, 2, 1)
sns.countplot(x='sex',data = benign_df)
plt.title('Benign')

plt.subplot(1, 2, 2)
sns.countplot(x='sex',data = malignant_df)
plt.title('Malignant')

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

plt.subplot(1, 2, 1)
sns.countplot(x='sex',data = benign_df)
plt.title('Benign')

plt.subplot(1, 2, 2)
sns.countplot(x='sex',data = malignant_df)
plt.title('Malignant')

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

plt.subplot(1, 2, 1)
sns.countplot(x='age_approx',data = benign_df)
plt.title('Benign')

plt.subplot(1, 2, 2)
sns.countplot(x='age_approx',data = malignant_df)
plt.title('Malignant')

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

plt.subplot(1, 2, 1)
sns.countplot(x='anatom_site_general_challenge',data = benign_df)
plt.title('Benign')

plt.subplot(1, 2, 2)
sns.countplot(x='anatom_site_general_challenge',data = malignant_df)
plt.title('Malignant')

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

plt.subplot(1, 2, 1)
sns.countplot(x='diagnosis',data = benign_df)
plt.title('Benign')

plt.subplot(1, 2, 2)
sns.countplot(x='diagnosis',data = malignant_df)
plt.title('Malignant')

plt.show()
# validation_split = 0.2
# train['image_name'] = train['image_name'].apply(lambda x: x+'.jpg') 

# train,valid = train_test_split(train[['image_name','benign_malignant']],test_size=validation_split,stratify=train['target'])

# print(np.mean(train['benign_malignant'].values))
# print(np.mean(valid['benign_malignant'].values))
if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except _:
            print("failed to initialize TPU")
    else:
        DEVICE = "GPU"

if DEVICE != "TPU":
    print("Using default strategy for CPU and single GPU")
    strategy = tf.distribute.get_strategy()

if DEVICE == "GPU":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    

AUTO     = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
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
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), 
                 K.dot(zoom_matrix,     shift_matrix))


def transform(image, cfg):    
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = cfg["read_size"]
    XDIM = DIM%2 #fix for size 331
    
    rot = cfg['rot'] * tf.random.normal([1], dtype='float32')
    shr = cfg['shr'] * tf.random.normal([1], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['hzoom']
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['wzoom']
    h_shift = cfg['hshift'] * tf.random.normal([1], dtype='float32') 
    w_shift = cfg['wshift'] * tf.random.normal([1], dtype='float32') 

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z   = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))
        
    return tf.reshape(d,[DIM, DIM,3])

def read_labeled_tfrecord(example):
    tfrec_format = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'target'                       : tf.io.FixedLenFeature([], tf.int64)
    }           
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'],example['target']


def read_unlabeled_tfrecord(example, return_image_name):
    tfrec_format = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['image_name'] if return_image_name else 0

def prepare_image(img, cfg=None, augment=True):    
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [cfg['read_size'], cfg['read_size']])
    img = tf.cast(img, tf.float32) / 255.0
    
    if augment:
        img = transform(img, cfg)
        img = tf.image.random_crop(img, [cfg['crop_size'], cfg['crop_size'], 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)

    else:
        img = tf.image.central_crop(img, cfg['crop_size'] / cfg['read_size'])
                                   
    img = tf.image.resize(img, [cfg['net_size'], cfg['net_size']])
    img = tf.reshape(img, [cfg['net_size'], cfg['net_size'], 3])
    return img

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
         for filename in filenames]
    return np.sum(n)

def get_dataset(files, cfg, augment = False, shuffle = False, repeat = False, 
                labeled=True, return_image_names=True,return_validation=False):
    
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()
    
    if repeat:
        ds = ds.repeat()
    
    if shuffle: 
        ds = ds.shuffle(1024*8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
        
    if labeled: 
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names), 
                    num_parallel_calls=AUTO)      
    
    ds = ds.map(lambda img, imgname_or_label: (prepare_image(img, augment=augment, cfg=cfg), 
                                               imgname_or_label), 
                num_parallel_calls=AUTO)
    
    if return_validation:
        val_ds = ds.take(int(count_data_items(files_train)*CFG['validation_split']))
        
        ds = ds.batch(cfg['batch_size'] * REPLICAS)
        val_ds =  val_ds.batch(cfg['batch_size'] * REPLICAS)
        ds = ds.prefetch(AUTO)
        val_ds = val_ds.prefetch(AUTO)
        
        return ds, val_ds

        
    
    ds = ds.batch(cfg['batch_size'] * REPLICAS)
    ds = ds.prefetch(AUTO)
    return ds

ds_train, ds_valid     = get_dataset(files_train, CFG, augment=True, shuffle=True, repeat=True,return_validation=True)
# ds_train     = ds_train.map(lambda img, label: (img, [label]))
steps_train  = int(count_data_items(files_train)*(1-CFG['validation_split'])) / (CFG['batch_size'] * REPLICAS)
steps_valid = int(count_data_items(files_train)*CFG['validation_split']) / (CFG['batch_size'] * REPLICAS)

print(steps_train,steps_valid)
pos = len(train[train['benign_malignant'] == 'malignant'])
neg = len(train[train['benign_malignant'] == 'benign'])


def make_model():
    with strategy.scope():
        initial_bias = np.log([pos/neg])
        output_bias = tf.keras.initializers.Constant(initial_bias)

        base_model = tf.keras.applications.ResNet152V2(
            include_top=False,
            weights="imagenet",
            input_shape = (CFG['net_size'],CFG['net_size'],3)
            )
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        predictions = keras.layers.Dense(1, activation='sigmoid',bias_initializer=output_bias)(x)

        model = keras.models.Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False
    
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy',tf.keras.metrics.AUC(name='auc')]
        )
    
        print(model.summary())
 
        return model

model = make_model()
def get_lr_callback(cfg):
    lr_start   = cfg['LR_START']
    lr_max     = cfg['LR_MAX'] * strategy.num_replicas_in_sync
    lr_min     = cfg['LR_MIN']
    lr_ramp_ep = cfg['LR_RAMPUP_EPOCHS']
    lr_sus_ep  = cfg['LR_SUSTAIN_EPOCHS']
    lr_decay   = cfg['LR_EXP_DECAY']
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback

es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',patience=5, verbose=1
)

rlr = get_lr_callback(CFG)
hist = model.fit(ds_train,verbose =1,validation_data=ds_valid,validation_steps=steps_valid,steps_per_epoch  = steps_train, epochs= CFG['epochs'],callbacks=[es,rlr])
CFG['batch_size'] = 128

cnt_test   = count_data_items(files_test)
steps      = cnt_test / (CFG['batch_size'] * REPLICAS)
ds_testAug = get_dataset(files_test, CFG, augment=True, repeat=True, 
                         labeled=False, return_image_names=False)
print(steps)
preds = model.predict(ds_testAug,steps=steps,verbose=5)
ds = get_dataset(files_test, CFG, augment=False, repeat=True, 
                 labeled=False, return_image_names=True)

image_names = []
it = iter(ds.unbatch())
for i in range(cnt_test):
    image_names.append(it.next()[1].numpy().decode("utf-8"))
    
image_names = np.array(image_names).flatten()


preds = preds.squeeze()[:len(image_names)]
submission = pd.DataFrame(dict(
    image_name = image_names,
    target     = preds.squeeze()))

submission = submission.sort_values('image_name')
submission.to_csv('first_submission.csv',index=False)
for i in hist.history.items():
    print(i[0]) 
import plotly.express as px

plot_df = pd.DataFrame(hist.history)
plot_df['epoch'] = np.arange(1,len(plot_df)+1)
print(plot_df)
fig = px.line(plot_df,x='epoch',y='val_accuracy')
fig.show()
submission.to_csv('submission.csv',index=False)

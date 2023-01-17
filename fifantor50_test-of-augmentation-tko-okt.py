DEVICE = "TPU"

CFG = dict(
    epochs = 50,
    lr = 0.1,  # learning rate  
    inp_size = 128, # input image size
    net_count         =   7,
    batch_size        =  16,
    
    read_size         = 128, 
    crop_size         = 128, 
    net_size          = 128,
    rot               = 180.0,
    shr               =   2.0,
    hzoom             =   8.0,
    wzoom             =   8.0,
    hshift            =   8.0,
    wshift            =   8.0,

    optimizer         = 'adam',
    label_smooth_fac  =   0.05,
    
    tta_steps         =  4  
)
!pip install -q efficientnet
import os, random, re, math, time
random.seed(a=42)

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
!pip install -q efficientnet
import efficientnet.tfkeras as efn

import PIL

from kaggle_datasets import KaggleDatasets

from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
SEED=42

def seed_everything(SEED):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

seed_everything(SEED)

BASEPATH = "../input/siim-isic-melanoma-classification"
df_train = pd.read_csv(os.path.join(BASEPATH, 'train.csv'))
df_test  = pd.read_csv(os.path.join(BASEPATH, 'test.csv'))
df_sub   = pd.read_csv(os.path.join(BASEPATH, 'sample_submission.csv'))


GCS_PATH    = KaggleDatasets().get_gcs_path('melanoma-128x128')
GCS_PATH1=KaggleDatasets().get_gcs_path('malignant-v2-128x128')
#GCS_PATH_2019    = KaggleDatasets().get_gcs_path('isic2019-384x384-cc')

#files_train_2019 = np.array(tf.io.gfile.glob(GCS_PATH_2019 + '/train*.tfrec'))
#files_train_2020 = np.array(tf.io.gfile.glob(GCS_PATH + '/train*.tfrec'))

files_train = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')
malignant= tf.io.gfile.glob(GCS_PATH1 + '/train*.tfrec')
files_train += malignant

#files_valid= tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')
#files_valid+=files_valid[:3]

#files_valid+=malignant

#files_train+=GCS_PATH_2019 
#files_train =files_train[3:]

#VALIDATION_FILENAMES1=tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')
#VALIDATION_FILENAMES1=VALIDATION_FILENAMES1[:3]

#VALIDATION_FILENAMES= tf.io.gfile.glob(GCS_PATH1 + '/train*.tfrec')
#VALIDATION_FILENAMES+=VALIDATION_FILENAMES1

#files_valid=VALIDATION_FILENAMES
files_test  = np.sort(np.array(tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')))

#files_train = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')
files_train ,files_valid = train_test_split(files_train,test_size = 0.20,random_state = SEED)

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
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),
        'sex'                          : tf.io.FixedLenFeature([], tf.int64),
        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        #'diagnosis'                    : tf.io.FixedLenFeature([], tf.int64),
        'target'                       : tf.io.FixedLenFeature([], tf.int64)
    }           
    example = tf.io.parse_single_example(example, tfrec_format)
    return (example['image'], (example['sex'], example['age_approx'], example['anatom_site_general_challenge']), example['target'])

def read_unlabeled_tfrecord(example, return_image_name):
    tfrec_format = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),
        'sex'                          : tf.io.FixedLenFeature([], tf.int64),
        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return (example['image'], (example['sex'], example['age_approx'], example['anatom_site_general_challenge']), (example['image_name'] if return_image_name else 0))

def prepare_data(data, cfg=None, augment=True):
    img = data[0]
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
    
    sex_oh = tf.one_hot(data[1][0], 2)
    age_aprox = tf.dtypes.cast(tf.reshape(data[1][1], [1]), tf.float32)
    anatom_site_general_challenge = tf.one_hot(data[1][2], 7)
    dense = tf.concat([sex_oh, age_aprox, anatom_site_general_challenge], axis=0)
    return (img, dense)

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
         for filename in filenames]
    return np.sum(n)

NUM_TEST_IMAGES = count_data_items(files_valid)
print(NUM_TEST_IMAGES)
def get_dataset(files, cfg, augment = False, shuffle = False, repeat = False, 
                labeled=True, return_image_names=True):
    
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

    ds = ds.map(lambda img, dense, imgname_or_label: (prepare_data((img, dense), augment=augment, cfg=cfg), 
                                               imgname_or_label), 
                num_parallel_calls=AUTO)
    
    ds = ds.batch(cfg['batch_size'] * REPLICAS)
    ds =ds.prefetch(AUTO)
    return ds
def show_dataset(thumb_size, cols, rows, ds):
    mosaic = PIL.Image.new(mode='RGB', size=(thumb_size*cols + (cols-1), 
                                             thumb_size*rows + (rows-1)))
   
    for idx, data in enumerate(iter(ds)):
        img, target_or_imgid = data[0][0], data[1]
        #img, target_or_imgid = data
        ix  = idx % cols
        iy  = idx // cols
        img = np.clip(img.numpy() * 255, 0, 255).astype(np.uint8)
        img = PIL.Image.fromarray(img)
        img = img.resize((thumb_size, thumb_size), resample=PIL.Image.BILINEAR)
        mosaic.paste(img, (ix*thumb_size + ix, 
                           iy*thumb_size + iy))

    display(mosaic)
    
ds = get_dataset(files_train, CFG).unbatch().take(12*5)   
show_dataset(64, 12, 5, ds)
ds = get_dataset(files_valid, CFG).unbatch().take(12*5)   
show_dataset(64, 12, 5, ds)
len(df_train.diagnosis.unique())
print("image before the augmentaion ")
ds = get_dataset(files_train, CFG).unbatch().take(1)   
show_dataset(200, 1, 1, ds)
print("image after the augmentaion ")
ds = tf.data.TFRecordDataset(files_train, num_parallel_reads=AUTO)
ds = ds.take(1).cache().repeat()
ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
# ds = ds.map(lambda img, target: (prepare_image(img, cfg=CFG, augment=True), target), 
#             num_parallel_calls=AUTO)
ds = ds.map(lambda img, dense, target: (prepare_data((img, dense), cfg=CFG, augment=True), target), 
            num_parallel_calls=AUTO)
ds = ds.take(12*5)
ds = ds.prefetch(AUTO)

show_dataset(64, 12, 5, ds)
ds = get_dataset(files_test, CFG, augment=True, repeat=True, 
                         labeled=False, return_image_names=False).unbatch().take(12*5)   
show_dataset(64, 12, 5, ds)
"""
def get_cosine_schedule_with_warmup(lr,num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lrfn(epoch):
        if epoch < num_warmup_steps:
            return (float(epoch) / float(max(1, num_warmup_steps))) * lr
        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

lr_schedule= get_cosine_schedule_with_warmup(lr=LR,num_warmup_steps=WARMUP,num_training_steps=EPOCHS)
"""
steps_train  = count_data_items(files_train) / (CFG['batch_size'] * REPLICAS)
steps_train =int(steps_train)
import kernel_tensorflow_utils as ktu
lr_callback = ktu.LRSchedulers.FineTuningLR(
    
    lr_start=1e-5, lr_max=5e-5 * strategy.num_replicas_in_sync, lr_min=1e-5,
    lr_rampup_epochs=5, lr_sustain_epochs=0, lr_exp_decay=0.8, verbose=1)

plt.figure(figsize=(8, 5))
lr_callback.visualize(steps_per_epoch=steps_train, epochs=40)
plt.figure(figsize=(8, 5))
lr_callback.visualize(steps_per_epoch=steps_train, epochs=40)
def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed
from keras import backend as K
import tensorflow as tf

def KerasFocalLoss(target, input):
    
    gamma = 2.
    input = tf.cast(input, tf.float32)
    
    max_val = K.clip(-input, 0, 1)
    loss = input - input * target + max_val + K.log(K.exp(-max_val) + K.exp(-input - max_val))
    invprobs = tf.log_sigmoid(-input * (target * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss
    
    return K.mean(K.sum(loss, axis=1))
from keras import backend as K
import tensorflow as tf

# Compatible with tensorflow backend

def focal_loss_f(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed
def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy
from keras.models import Sequential, load_model
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate,multiply, LocallyConnected2D, Lambda)
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
import keras
from keras.models import Model
from keras.activations import hard_sigmoid
x = tf.constant([-3.0,-1.0, 0.0,1.0,3.0], dtype = tf.float32)
y = tf.keras.backend.hard_sigmoid(x)
y.numpy()
from keras.activations import hard_sigmoid
def EFFB_7():
    with strategy.scope():
        
        # meta data:
        # https://www.kaggle.com/rajnishe/rc-fork-siim-isic-melanoma-384x384/notebook
        
        img_inp = tf.keras.layers.Input(shape = (CFG['inp_size'], CFG['inp_size'], 3), name = 'img_inp')
        meta_inp = tf.keras.layers.Input(shape = (10), name = 'meta_inp')
        eff ='6'
        constructor = getattr(efn, f'EfficientNetB{eff}')
        efnetb = constructor(weights = 'noisy-student', include_top = False)     
        
        # attention :
        # https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age/notebook
        
        pt_depth = efnetb.get_output_shape_at(0)[-1]
        pt_features = efnetb(img_inp)
        bn_features = tf.keras.layers.BatchNormalization()(pt_features)
        attn_layer = tf.keras.layers.Conv2D(64, kernel_size = (1, 1), padding = 'same', activation = 'swish')(tf.keras.layers.Dropout(0.5)(bn_features))
        attn_layer = tf.keras.layers.Conv2D(16, kernel_size = (1, 1), padding = 'same', activation = 'swish')(attn_layer)
        attn_layer = tf.keras.layers.Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'swish')(attn_layer)
        attn_layer = tf.keras.layers.Conv2D(1, kernel_size = (1, 1), padding = 'valid', activation = hard_sigmoid)(attn_layer)
        up_c2_w = np.ones((1, 1, 1, pt_depth))
        up_c2 = tf.keras.layers.Conv2D(pt_depth, kernel_size = (1, 1), padding = 'same',  activation = 'linear',  use_bias = False,    weights = [up_c2_w]  )
        up_c2.trainable = False
        attn_layer = up_c2(attn_layer)
        mask_features = tf.keras.layers.multiply([attn_layer, bn_features])
        gap_features = tf.keras.layers.GlobalAveragePooling2D()(mask_features)
        gap_mask = tf.keras.layers.GlobalAveragePooling2D()(attn_layer)
        
        gap = tf.keras.layers.Lambda(lambda x: x[0] / x[1], name = 'RescaleGAP')([gap_features, gap_mask])
        gap_dr = tf.keras.layers.Dropout(0.5)(gap)
        dr_steps = tf.keras.layers.Dropout(0.5)(tf.keras.layers.Dense(128, activation = 'swish')(gap_dr))
        
        meta_layer = tf.keras.layers.Dense(16)(meta_inp)
        meta_layer = tf.keras.layers.BatchNormalization()(meta_layer)
        meta_layer = tf.keras.layers.Activation('swish')(meta_layer)
        meta_layer = tf.keras.layers.Dropout(0.5)(meta_layer)
        meta_layer = tf.keras.layers.Dense(8)(meta_inp)
        meta_layer = tf.keras.layers.BatchNormalization()(meta_layer)
        meta_layer = tf.keras.layers.Activation('swish')(meta_layer)
        meta_layer = tf.keras.layers.Dropout(0.5)(meta_layer)
        
        concat = tf.keras.layers.concatenate([dr_steps, meta_layer])
        concat = tf.keras.layers.BatchNormalization()(concat)
        concat = tf.keras.layers.Dense(512, activation = 'swish')(concat)        
        concat = tf.keras.layers.Dropout(0.5)(concat)
        output = tf.keras.layers.Dense(2, activation ='softmax',dtype='float32')(concat)

        model = tf.keras.models.Model(inputs = [img_inp, meta_inp], outputs = [output])

        return model
import tensorflow_addons as tfa
opt = tfa.optimizers.RectifiedAdam()

#tfa.losses.WeightedKappa(num_classes=1)
INI_LR=1.e-4
import tensorflow.keras as K
#opt = K.optimizers.Adam(lr=INI_LR)
import tensorflow as tf
from keras.losses import binary_crossentropy, categorical_crossentropy
import keras.backend as K
import numpy as np
from prettytable import PrettyTable
from prettytable import ALL
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def fbeta_score_macro(y_true, y_pred, beta=1, threshold=0.1):

    y_true = K.cast(y_true, 'float')
    y_pred = K.cast(K.greater(K.cast(y_pred, 'float'), threshold), 'float')

    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = (1 + beta ** 2) * p * r / ((beta ** 2) * p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
def f1_loss(predict, target):
    loss = 0
    lack_cls = target.sum(dim=0) == 0
    if lack_cls.any():
        loss += F.binary_cross_entropy_with_logits(
            predict[:, lack_cls], target[:, lack_cls])
    predict = torch.sigmoid(predict)
    predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (target.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    return 1 - f1.mean() + loss
opt = tfa.optimizers.RectifiedAdam()
ranger = tfa.optimizers.Lookahead(opt, 
                                  sync_period=6, 
                                  slow_step_size=0.5)
def compile_new_model():    
    with strategy.scope():
        model=EFFB_7()
        
        # warm up model
        #for layer in model.layers:
           # layer.trainable = False

        #for i in range(-3,0):
            #model.layers[i].trainable = True
            
         # train all layers
        #for layer in model.layers:
            #layer.trainable = True 
        
        #opt = tfa.optimizers.RectifiedAdam()
        model.compile(
            optimizer ='adam',
            #loss = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO),
            loss = 'sparse_categorical_crossentropy',
            # metrics=['sparse_categorical_accuracy']
            metrics = ['sparse_categorical_accuracy']
        )

        
    return model
from keras import backend as K
#df_category = pd.merge(train2019 , train , on="target")
"""
train = pd.read_csv(os.path.join(BASEPATH, 'train.csv'))
print('train: ', train.shape, '| unique ids:', sum(train['target'].value_counts()))
X_train, X_val = train_test_split(train, test_size=.2, stratify=train['target'], random_state=SEED)
"""
"""
lbl_value_counts = train['target'].value_counts()
class_weights = {i: max(lbl_value_counts) / v for i, v in lbl_value_counts.items()}
print('classes weigths:', class_weights)
"""
from sklearn.utils import class_weight
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train.target),
                                                 train.target)
class_weights = dict(enumerate(class_weights))

#class_weights
file_path="/kaggle/working/effnet6_weights.best.hdf5"

checkpoint = ModelCheckpoint(file_path, monitor='auc', verbose=1, save_best_only=True, mode='max')
ds_train     = get_dataset(files_train, CFG, augment=True, shuffle=True, repeat=True)
steps_train  = count_data_items(files_train) / (CFG['batch_size'] * REPLICAS)
#ds_train     = ds_train.map(lambda img, label: (img, tuple([label] * CFG['net_count'])))
#ds_train     = ds_train.map(lambda img, label: (img, tuple([label] * CFG['net_count'])))
#import tensorflow as tf
#train_set = tf.data.Dataset.from_tensor_slices(ds_train)
#ds_t  = ds_train[ds_train['target'] == 0]
#tf.data.Dataset.from_tensor_slices(list(ds_train))
"""
# Class count

count_class_0, count_class_1 = df_train.target.value_counts()

# Divide by class
df_class_0 = df_train[df_train['target'] == 0]
df_class_1 = df_train[df_train['target'] == 1]
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over.target.value_counts())

df_test_over.target.value_counts().plot(kind='bar', title='Count (target)');
"""
print("Buidling model...")
model = compile_new_model()
model.summary()
"""
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
"""
!pip install livelossplot
from livelossplot import PlotLossesKeras
cb=[PlotLossesKeras()]
ds_valid= get_dataset(files_valid , CFG, augment=True, shuffle=True, repeat=True)
steps_valid= count_data_items(files_valid) / (CFG['batch_size']*8)
print(steps_valid)
WORKERS=2
#class_weight=class_weights
with strategy.scope():
    history      = model.fit(ds_train, 
                                 verbose          = 1,
                                 steps_per_epoch  = steps_train, 
                                 epochs           = CFG['epochs'],
                                 callbacks        = [lr_callback ,cb],
                                 validation_data  = ds_valid,
                                 workers=WORKERS, use_multiprocessing=True,
                                 #class_weight=class_weights,
                                 validation_steps  = steps_valid
                                 )
#model = compile_new_model()
#model.load_weights(file_path)
steps_valid= count_data_items(files_valid)
print(steps_valid)
CFG['batch_size'] = 256

cnt_test   = count_data_items(files_valid)
steps      = cnt_test / (CFG['batch_size'] * REPLICAS) * CFG['tta_steps']
ds_testAug = get_dataset(files_valid, CFG, augment=True, repeat=True, 
                         labeled=False, return_image_names=False)

probs = model.predict(ds_testAug, verbose=1, steps=steps)
probs = np.stack(probs)
probs = probs2 = probs[:cnt_test * CFG['tta_steps'],0]
probs = np.stack(np.split(probs, CFG['tta_steps'], axis=0), axis=1)
probs = np.mean(probs, axis=1)
cmdataset = get_dataset(files_valid, CFG, augment=True, repeat=True ,labeled=True, return_image_names=False) # since we are splitting the dataset and iterating separately on images and labels, order matters.
images_ds = cmdataset.map(lambda image, label: image)
labels_ds = cmdataset.map(lambda image, label: label).unbatch()
cm_correct_labels = next(iter(labels_ds.batch(NUM_TEST_IMAGES))).numpy() # get everything as one batch
#cm_probabilities = model.predict(images_ds)
#cm_predictions = np.argmax(cm_probabilities, axis=-1)
print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)
print("Predicted labels: ", probs, probs)
len(probs)
from sklearn.metrics import confusion_matrix
import sklearn.metrics
y_true =cm_correct_labels
y_pred = probs
tn, fp, fn, tp = confusion_matrix(y_true, y_pred.round()).ravel()
specificity = tn /(tn+fp)
sensitivity=  tp/ (tp+fn)
Precision = tp/(tp+fp)
Recall = tp/ (tp+fn)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)

print('Specificity : {:.3f}, Sensitivity: {:.3f}, F1_Score: {:.3f}'.format(specificity, sensitivity,F1_Score))
import sklearn.metrics
print(sklearn.metrics.classification_report(y_true, y_pred.round()))
c_m=confusion_matrix(y_true, y_pred.round())
print(c_m)
print('Specificity : {:.3f}, Sensitivity: {:.3f}, F1_Score: {:.3f}'.format(specificity, sensitivity,F1_Score))
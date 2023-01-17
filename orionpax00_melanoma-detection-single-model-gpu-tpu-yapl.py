!pip install yapl==0.1.2 efficientnet > /dev/null

!rm -rf *
import yapl

import os

import re

import random

import pandas as pd

import numpy as np

import math



from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import tensorflow as tf

import tensorflow.keras.backend as K

from kaggle_datasets import KaggleDatasets

import efficientnet.tfkeras as efn



from yapl.config.config import Config

import matplotlib.pyplot as plt
tf.keras.backend.clear_session()
class configSetup(Config):

    def __init__(self):

        super().__init__()

        self.IS_EXT = True

        self.DO_VAL_SPLIT = True

        self.EXPERIMENT_NAME = 'melanoma-detection-yapl-testing'

        

        if self.IS_EXT:

            self.GCS_DS_PATH = KaggleDatasets().get_gcs_path('melanoma-384x384')

            self.TOTAL_FILES = tf.io.gfile.glob(self.GCS_DS_PATH + '/train*.tfrec')

            self.TEST_FILES = tf.io.gfile.glob(self.GCS_DS_PATH + '/test*.tfrec')

            self.TRAIN_CSV = '../input/melanoma-128x128/train.csv'

        else:

            self.GCS_DS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

            self.TOTAL_FILES = tf.io.gfile.glob(self.GCS_DS_PATH + '/train/*.tfrec')

            self.TEST_FILES = tf.io.gfile.glob(self.GCS_DS_PATH + '/test/*.tfrec')

            self.TRAIN_CSV = '../input/siim-isic-melanoma-classification/train.csv'

        

        self.TEST_CSV = '../input/siim-isic-melanoma-classification/test.csv'

        

        self.VALIDATION_CSV = ""

        

        self.TOTAL_TRAIN_IMG = 0

        self.TOTAL_TEST_IMG = 0

        self.NEG_SMAPLE = 32542

        self.POS_SMAPLE = 584

        

        self.IMG_SIZE = [384, 384]

        self.IMG_TRAIN_SHAPE = [384,384]

        self.IMG_SHAPE = (384, 384, 3)

        self.DO_FINETUNE = True

        

        self.BATCH_SIZE = 16 # 16

        self.EPOCHES = 10 

        self.SEED = 127

        self.FOLDS = 3

        

        self.LOSS = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.05)

        self.OPTIMIZER = tf.keras.optimizers.Adam()

        self.ACCURACY = ['AUC']

        

        self.CALLBACKS = [

            tf.keras.callbacks.EarlyStopping(

                monitor='val_auc', restore_best_weights=True,  mode = 'max', patience = 3, verbose = 1, min_delta = 0.001,

            )

        ]

        

        self.STRATEGY = None
# store config in yapl global variable        

configSetup().make_global()



yapl.backend = 'tf'



# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    yapl.config.STRATEGY = strategy

    yapl.config.BATCH_SIZE = 32 * strategy.num_replicas_in_sync 

else:

    strategy = tf.distribute.get_strategy() 



#configuration

# yapl.config.TOTAL_TRAIN_IMG = len(pd.read_csv(yapl.config.TRAIN_CSV).image_name)

yapl.config.TOTAL_TEST_IMG = len(pd.read_csv(yapl.config.TEST_CSV).image_name)





def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
train_df = pd.read_csv(yapl.config.TRAIN_CSV)

train_df.head()
FOLDS_DICT = {}



skf = KFold(n_splits=yapl.config.FOLDS,shuffle=True,random_state=yapl.config.SEED)

for fold,(idxT,idxV) in enumerate(skf.split(np.arange(15))):

    FOLDS_DICT['fold_{}'.format(fold+1)] = {

                                            "trainfiles" : [yapl.config.TOTAL_FILES[x] for x in idxT],

                                            "valfiles"   : [yapl.config.TOTAL_FILES[x] for x in idxV]

                                            }
def seed_everything():

    np.random.seed(yapl.config.SEED)

    tf.random.set_seed(yapl.config.SEED)

    random.seed(a=yapl.config.SEED)

    os.environ['PYTHONHASHSEED'] = str(yapl.config.SEED)



seed_everything()
ROT_ = 180.0

SHR_ = 2.0

HZOOM_ = 8.0

WZOOM_ = 8.0

HSHIFT_ = 8.0

WSHIFT_ = 8.0



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





def transform(image, DIM=256):    

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, sheared, zoomed, and shifted

    XDIM = DIM%2 #fix for size 331

    

    rot = ROT_ * tf.random.normal([1], dtype='float32')

    shr = SHR_ * tf.random.normal([1], dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / HZOOM_

    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / WZOOM_

    h_shift = HSHIFT_ * tf.random.normal([1], dtype='float32') 

    w_shift = WSHIFT_ * tf.random.normal([1], dtype='float32') 



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
## Helper Functions

def data_augment(image, label):

    image = transform(image, DIM=yapl.config.IMG_SIZE[0])

    image = tf.image.random_flip_left_right(image)

    image = tf.image.rot90(image)

    

    return image, label  



def process_training_data(data_file):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), 

        "target": tf.io.FixedLenFeature([], tf.int64),

    }

    data = tf.io.parse_single_example(data_file, LABELED_TFREC_FORMAT)

    img = tf.image.decode_jpeg(data['image'], channels=3)

#     img = tf.image.resize(img, yapl.config.IMG_TRAIN_SHAPE)

    img = tf.cast(img, tf.float32) / 255.0

    img = tf.reshape(img, [*yapl.config.IMG_TRAIN_SHAPE, 3])



    label = tf.cast(data['target'], tf.int32)



    return img, label



def process_test_data(data_file):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "image_name": tf.io.FixedLenFeature([], tf.string),

    }

    data = tf.io.parse_single_example(data_file, LABELED_TFREC_FORMAT)

    img = tf.image.decode_jpeg(data['image'], channels=3)

#     img = tf.image.resize(img, yapl.config.IMG_TRAIN_SHAPE)

    img = tf.cast(img, tf.float32) / 255.0

    img = tf.reshape(img, [*yapl.config.IMG_TRAIN_SHAPE, 3])



    idnum = data['image_name']



    return img, idnum
#learning rate Scheduler

def build_lrfn(lr_start=0.000005, lr_max=0.000020, 

               lr_min=0.000001, lr_rampup_epochs=5, 

               lr_sustain_epochs=0, lr_exp_decay=.8):

    if yapl.config.STRATEGY is not None:

        lr_max = lr_max * yapl.config.STRATEGY.num_replicas_in_sync



    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    

    return lrfn



#learning rate Scheduler

def build_lrfn_l(lr_start=0.00001, lr_max=0.000075, 

               lr_min=0.000001, lr_rampup_epochs=20, 

               lr_sustain_epochs=0, lr_exp_decay=.8):

    if yapl.config.STRATEGY is not None:

        lr_max = lr_max * yapl.config.STRATEGY.num_replicas_in_sync



    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    

    return lrfn



def build_lrfn_2():

    lr_start   = 0.000005

    lr_max     = 0.000020 

    lr_min     = 0.000001

    lr_ramp_ep = 5

    lr_sus_ep  = 0

    lr_decay   = 0.8

    if yapl.config.STRATEGY is not None:

        lr_max = lr_max * yapl.config.BATCH_SIZE

   

    def lrfn(epoch):

        if epoch < lr_ramp_ep:

            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

            

        elif epoch < lr_ramp_ep + lr_sus_ep:

            lr = lr_max

            

        else:

            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min

            

        return lr



    return lrfn





def get_cosine_schedule_with_warmup(lr = 0.00004, num_warmup_steps = 5 , num_training_steps = yapl.config.EPOCHES, num_cycles=0.5):

    def lrfn(epoch):

        if epoch < num_warmup_steps:

            return (float(epoch) / float(max(1, num_warmup_steps))) * lr

        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr



    return lrfn



lrfn = get_cosine_schedule_with_warmup()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=0)

yapl.config.CALLBACKS.append(lr_schedule)
def efficientnetbx():

    return tf.keras.Sequential([

                    efn.EfficientNetB0(

                        input_shape=(*yapl.config.IMG_TRAIN_SHAPE, 3),

                        weights='imagenet',

                        include_top=False

                    ),

                    tf.keras.layers.GlobalAveragePooling2D(),

                    tf.keras.layers.Dense(1, activation='sigmoid')

                ])
def fitengine(model, traindataset, valdataset = None, istraining = True):

    model.compile(

        optimizer   =  yapl.config.OPTIMIZER, 

        loss        =  yapl.config.LOSS, 

        metrics     =  yapl.config.ACCURACY

    )

    weight_for_0 = (1 / yapl.config.NEG_SMAPLE)*(yapl.config.TOTAL_TRAIN_IMG)/2.0 

    weight_for_1 = (1 / yapl.config.POS_SMAPLE)*(yapl.config.TOTAL_TRAIN_IMG)/2.0



    class_weight = {0: weight_for_0, 1: weight_for_1}

    

    history = model.fit(

                traindataset, 

                epochs            =   yapl.config.EPOCHES, 

                steps_per_epoch   =   yapl.config.TOTAL_TRAIN_IMG//yapl.config.BATCH_SIZE,

                callbacks         =   yapl.config.CALLBACKS,

                validation_data   =   valdataset,

                validation_steps  =   yapl.config.TOTAL_VAL_IMG//yapl.config.BATCH_SIZE,

                verbose           =   0

            )



    return history
MODELS = {}

HISTORY = {}

TTA = 11



print('##'*30)

print("#### IMAGE SIZE {} ".format(yapl.config.IMG_SHAPE))

print("#### BATCH SIZE {} ".format(yapl.config.BATCH_SIZE))

print("#### TRAINING RECORDS {} ".format(15 - 15/yapl.config.FOLDS))

print("#### VALIDATION RECORDS {} ".format(15/yapl.config.FOLDS))

print('##'*30)

print("\n\n")



for fold in range(yapl.config.FOLDS):

    K.clear_session()

    yapl.config.CALLBACKS[0] = tf.keras.callbacks.ModelCheckpoint(

                                    'fold-%i.h5'%(fold+1), monitor='val_loss', verbose=0, save_best_only=True,

                                    save_weights_only=True, mode='min', save_freq='epoch')

    

    yapl.config.TOTAL_TRAIN_IMG = count_data_items(FOLDS_DICT['fold_{}'.format(fold+1)]['trainfiles'])

    yapl.config.TOTAL_VAL_IMG = count_data_items(FOLDS_DICT['fold_{}'.format(fold+1)]['valfiles'])

        

    # training data

    ignore_order = tf.data.Options()

    ignore_order.experimental_deterministic = False

    train_dataset = (

        tf.data.TFRecordDataset(

            FOLDS_DICT['fold_{}'.format(fold+1)]['trainfiles'],  

            num_parallel_reads=tf.data.experimental.AUTOTUNE

        ).with_options(

            ignore_order

        ).map(

            process_training_data,

            num_parallel_calls=tf.data.experimental.AUTOTUNE

        ).map(

            data_augment, 

            num_parallel_calls=tf.data.experimental.AUTOTUNE

        ).repeat(

        ).shuffle(

            yapl.config.SEED

        ).batch(

            yapl.config.BATCH_SIZE

        ).prefetch(

            tf.data.experimental.AUTOTUNE

        )

    )



    #Validation data

    ignore_order = tf.data.Options()

    val_dataset = (

        tf.data.TFRecordDataset(

            FOLDS_DICT['fold_{}'.format(fold+1)]['valfiles'],  

            num_parallel_reads=tf.data.experimental.AUTOTUNE

        ).with_options(

            ignore_order

        ).map(

            process_training_data,

            num_parallel_calls=tf.data.experimental.AUTOTUNE

        ).batch(

            yapl.config.BATCH_SIZE

        ).prefetch(

            tf.data.experimental.AUTOTUNE

        )

    )



    if yapl.config.STRATEGY is not None:

        with strategy.scope():

            model = efficientnetbx()

    else:

        model = efficientnetbx()



    print("##"*30)

    print("#### FOLD {} ".format(fold+1))

    

    

    history = fitengine(model, train_dataset, val_dataset); #trining model

    HISTORY['fold_{}'.format(fold+1)] = history #storing history

    

    

    model.load_weights('fold-%i.h5'%(fold+1)) #loading saved model

    

    max_val_auc = np.max( history.history['val_auc'])

    print('#### Validation AUC without TTA: %.4f'%max_val_auc)

    

    ## Prediction Setup

    ignore_order = tf.data.Options()

    test_dataset = (

        tf.data.TFRecordDataset(

            yapl.config.TEST_FILES,  

            num_parallel_reads=tf.data.experimental.AUTOTUNE

        ).with_options(

            ignore_order

        ).map(

            process_test_data,

            num_parallel_calls=tf.data.experimental.AUTOTUNE

        ).map(

            data_augment, 

            num_parallel_calls=tf.data.experimental.AUTOTUNE

        ).repeat(

        ).batch(

            yapl.config.BATCH_SIZE * 4 

        ).prefetch(

            tf.data.experimental.AUTOTUNE

        )

    )      



    test_imgs = test_dataset.map(lambda images, ids: images)

    img_ids_ds = test_dataset.map(lambda images, ids: ids).unbatch()

        

    ct_test = count_data_items(yapl.config.TEST_FILES)

    STEPS = TTA * ct_test/(yapl.config.BATCH_SIZE * 4)

    pred = model.predict(test_imgs,steps=STEPS,verbose=0)[:TTA*ct_test,] 

    predictions = np.mean(pred.reshape((ct_test,TTA),order='F'),axis=1) 

    

    test_ids = next(iter(img_ids_ds.batch(yapl.config.TOTAL_TEST_IMG))).numpy().astype('U') 



    pd.DataFrame({

         'image_name'  : test_ids, 

         'target'      : predictions

        }).to_csv('prediction_fold_{}.csv'.format(fold+1), index=False)   

    

    

    #Predicting on validation Data with TTA

    ignore_order = tf.data.Options()

    val_dataset = (

        tf.data.TFRecordDataset(

            FOLDS_DICT['fold_{}'.format(fold+1)]['valfiles'], 

            num_parallel_reads=tf.data.experimental.AUTOTUNE

        ).with_options(

            ignore_order

        ).map(

            process_training_data,

            num_parallel_calls=tf.data.experimental.AUTOTUNE

        ).map(

            data_augment, 

            num_parallel_calls=tf.data.experimental.AUTOTUNE

        ).repeat(

        ).batch(

            yapl.config.BATCH_SIZE * 4

        ).prefetch(

            tf.data.experimental.AUTOTUNE

        )

    )

    

    test_imgs = val_dataset.map(lambda images, labels: images)

    img_labels_ds = val_dataset.map(lambda images, labels: labels).unbatch()

    

    ct_test = yapl.config.TOTAL_VAL_IMG

    STEPS = TTA * ct_test/(yapl.config.BATCH_SIZE * 4)

    pred = model.predict(test_imgs,steps=STEPS,verbose=0)[:TTA*ct_test,] 

    predictions = np.mean(pred.reshape((ct_test,TTA),order='F'),axis=1)

    

    test_labels = next(iter(img_labels_ds.batch(yapl.config.TOTAL_VAL_IMG))).numpy()



    pd.DataFrame(

        {

         'target'    : predictions,

         'label'     : test_labels

        }

    ).to_csv('oof_{}.csv'.format(fold+1), index=False)  

    

    

    ttavalauc = roc_auc_score(test_labels,predictions)

    print('#### Validation AUC with TTA: %.4f'%ttavalauc)

    

    print('##'*30)

    #Displaying Training

    plt.figure(figsize=(15,5))

    plt.plot(np.arange(yapl.config.EPOCHES),history.history['auc'],'-o',label='Train AUC',color='#ff7f0e')

    plt.plot(np.arange(yapl.config.EPOCHES),history.history['val_auc'],'-o',label='Val AUC',color='#1f77b4')

    x = np.argmax( history.history['val_auc'] ); y = np.max( history.history['val_auc'] )

    xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]

    plt.scatter(x,y,s=200,color='#1f77b4'); plt.text(x-0.03*xdist,y-0.13*ydist,'max auc\n%.2f'%y,size=14)

    plt.ylabel('AUC',size=14); plt.xlabel('Epoch',size=14)

    plt.legend(loc=2)

    plt2 = plt.gca().twinx()

    plt2.plot(np.arange(yapl.config.EPOCHES),history.history['loss'],'-o',label='Train Loss',color='#2ca02c')

    plt2.plot(np.arange(yapl.config.EPOCHES),history.history['val_loss'],'-o',label='Val Loss',color='#d62728')

    x = np.argmin( history.history['val_loss'] ); y = np.min( history.history['val_loss'] )

    ydist = plt.ylim()[1] - plt.ylim()[0]

    plt.scatter(x,y,s=200,color='#d62728'); plt.text(x-0.03*xdist,y+0.05*ydist,'min loss',size=14)

    plt.ylabel('Loss',size=14)

    plt.legend(loc=3)

    plt.show()
## Learning Rate plot

import matplotlib.pyplot as plt



lrfn = build_lrfn_2()



plt.plot([lrfn(epoch) for epoch in range(yapl.config.EPOCHES)])
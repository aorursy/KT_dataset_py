DEVICE = "TPU"



CFG = dict(

    epochs = 25,

    lr = 0.1,  # learning rate  

    inp_size = 512, # input image size

    net_count         =   7,

    batch_size        =  32,

    

    read_size         =512, 

    crop_size         = 512, 

    net_size          = 512,

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





GCS_PATH    = KaggleDatasets().get_gcs_path('512x512-melanoma-tfrecords-70k-images')

#GCS_PATH_2019    = KaggleDatasets().get_gcs_path('isic2019-128x128')



#files_train_2019 = np.array(tf.io.gfile.glob(GCS_PATH_2019 + '/train*.tfrec'))

#files_train_2020 = np.array(tf.io.gfile.glob(GCS_PATH + '/train*.tfrec'))



files_train = np.sort(np.array(tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')))

files_test  = np.sort(np.array(tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')))
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

    ds = ds.prefetch(AUTO)

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
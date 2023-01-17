!pip install -q efficientnet
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, random, re, math, time

random.seed(a=42)

import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow.keras.backend as K





from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D

from tensorflow.keras.layers import GlobalMaxPooling2D

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Concatenate

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import cv2

from tensorflow.keras import backend as K

import seaborn as sns

color = sns.color_palette()

%matplotlib inline



from skimage.io import imread

import efficientnet.tfkeras as efn

import PIL

from kaggle_datasets import KaggleDatasets

from tqdm import tqdm

import matplotlib.pyplot as  plt

import seaborn as sns
tf.__version__ 
BASEPATH = "../input/siim-isic-melanoma-classification"

df_train = pd.read_csv(os.path.join(BASEPATH, 'train.csv'))

df_test  = pd.read_csv(os.path.join(BASEPATH, 'test.csv'))

df_sub   = pd.read_csv(os.path.join(BASEPATH, 'sample_submission.csv'))

# Get the counts for each class

cases_count = df_train['target'].value_counts()

print(cases_count)



# Plot the results 

plt.figure(figsize=(10,8))

sns.barplot(x=cases_count.index, y= cases_count.values)

plt.title('Number of cases', fontsize=14)

plt.xlabel('Case type', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'melanoma(1)'])

plt.show()
GCS_PATH    = KaggleDatasets().get_gcs_path('melanoma-256x256')

GCS_PATH2    = KaggleDatasets().get_gcs_path('isic2019-256x256')

files_train = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')

#files_train += tf.io.gfile.glob(GCS_PATH2 + '/train*.tfrec')

files_train1 = tf.io.gfile.glob(GCS_PATH2 + '/train*.tfrec')

files_test  = np.sort(np.array(tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')))

#np.random.shuffle(files_train)
DEVICE = "TPU"

CFG = dict(

    batch_size        =  128,

    read_size         = 256, 

    crop_size         = 250, 

    net_size          = 248,

    LR_START          =   0.00005,

    LR_MAX            =   0.00020,

    LR_MIN            =   0.00001,

    LR_RAMPUP_EPOCHS  =   5,

    LR_SUSTAIN_EPOCHS =   0,

    LR_EXP_DECAY      =   0.8,

    

    epochs            =  5,

    

    rot               = 180.0,

    shr               =   1.5,

    hzoom             =   6.0,

    wzoom             =   6.0,

    hshift            =   6.0,

    wshift            =   6.0,

    optimizer         = 'adam',

    label_smooth_fac  =   0.05,

    tta_steps         =  25    

)
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

        'diagnosis'                    : tf.io.FixedLenFeature([], tf.int64),

        'target'                       : tf.io.FixedLenFeature([], tf.int64)

    }           

    example = tf.io.parse_single_example(example, tfrec_format)

    return example['image'], example['target']





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

    

    ds = ds.map(lambda img, imgname_or_label: (prepare_image(img, augment=augment, cfg=cfg), 

                                               imgname_or_label), 

                num_parallel_calls=AUTO)

    

    ds = ds.batch(cfg['batch_size'] * REPLICAS)

    ds = ds.prefetch(AUTO)

    return ds
# Get few samples for both the classes

shonenkovData = pd.read_csv('../input/melanoma-merged-external-data-512x512-jpeg/folds.csv')

melanoma_samples = (shonenkovData[shonenkovData['target']==1]['image_id'].iloc[:5]).tolist()

normal_samples = (shonenkovData[shonenkovData['target']==0]['image_id'].iloc[:5]).tolist()

# Concat the data in a single list and del the above two list

samples = melanoma_samples + normal_samples

del melanoma_samples, normal_samples

source = "../input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma/"

# Plot the data 

f, ax = plt.subplots(2,5, figsize=(30,10))

for i in range(10):

    img = imread(source + samples[i]+".jpg")

    ax[i//5, i%5].imshow(img, cmap='gray')

    if i<5:

        ax[i//5, i%5].set_title("melanoma")

    else:

        ax[i//5, i%5].set_title("Normal")

    ax[i//5, i%5].axis('off')

    ax[i//5, i%5].set_aspect('auto')

plt.show()
def show_dataset(thumb_size, cols, rows, ds):

    mosaic = PIL.Image.new(mode='RGB', size=(thumb_size*cols + (cols-1), 

                                             thumb_size*rows + (rows-1)))

   

    for idx, data in enumerate(iter(ds)):

        img, target_or_imgid = data

        ix  = idx % cols

        iy  = idx // cols

        img = np.clip(img.numpy() * 255, 0, 255).astype(np.uint8)

        img = PIL.Image.fromarray(img)

        img = img.resize((thumb_size, thumb_size), resample=PIL.Image.BILINEAR)

        mosaic.paste(img, (ix*thumb_size + ix, 

                           iy*thumb_size + iy))



    display(mosaic)

    

ds = get_dataset(files_train, CFG).unbatch().take(12*2)   

show_dataset(512, 12, 2, ds)
ds = tf.data.TFRecordDataset(files_train, num_parallel_reads=AUTO)

ds = ds.take(1).cache().repeat()

ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)

ds = ds.map(lambda img, target: (prepare_image(img, cfg=CFG, augment=True), target), 

            num_parallel_calls=AUTO)

ds = ds.take(12*2)

ds = ds.prefetch(AUTO)



show_dataset(512, 12, 2, ds)
ds = get_dataset(files_test, CFG, labeled=False).unbatch().take(12*2)   

show_dataset(512, 12, 2, ds)




# Get few samples for both the classes

chrisData = pd.read_csv('../input/jpeg-isic2019-256x256/train.csv')

melanoma_samples = (chrisData[chrisData['target']==1]['image_name'].iloc[:5]).tolist()

normal_samples = (chrisData[chrisData['target']==0]['image_name'].iloc[:5]).tolist()

# Concat the data in a single list and del the above two list

samples = melanoma_samples + normal_samples

del melanoma_samples, normal_samples

source = "../input/jpeg-isic2019-256x256/train/"

# Plot the data 

f, ax = plt.subplots(2,5, figsize=(30,10))

for i in range(10):

    img = imread(source + samples[i]+".jpg")

    ax[i//5, i%5].imshow(img, cmap='gray')

    if i<5:

        ax[i//5, i%5].set_title("melanoma")

    else:

        ax[i//5, i%5].set_title("Normal")

    ax[i//5, i%5].axis('off')

    ax[i//5, i%5].set_aspect('auto')

plt.show()
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
#https://www.kaggle.com/aakashnain/beating-everything-with-depthwise-convolution

def build_model():

    input_img = Input(shape=(256,256,3), name='ImageInput')

    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)

    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)

    x = MaxPooling2D((2,2), name='pool1')(x)

    

    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)

    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)

    x = MaxPooling2D((2,2), name='pool2')(x)

    

    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)

    x = BatchNormalization(name='bn1')(x)

    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)

    x = BatchNormalization(name='bn2')(x)

    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)

    x = MaxPooling2D((2,2), name='pool3')(x)

    

    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)

    x = BatchNormalization(name='bn3')(x)

    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)

    x = BatchNormalization(name='bn4')(x)

    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)

    x = MaxPooling2D((3,3), name='pool4')(x)

    

    x = Flatten(name='flatten')(x)





    dense = []

    for p in np.linspace(0.5,0.7, 5):

        x = Dense(1024, activation='relu', name= f'fc1{p}')(x)

        x_ = tf.keras.layers.Dropout(p)(x)

        dense.append(x_)

    x = tf.keras.layers.Average()(dense)

    



    dense = []

    for p in np.linspace(0.3,0.5, 5):

        x = Dense(512, activation='relu', name= f'fc2{p}')(x)

        x_ = tf.keras.layers.Dropout(p)(x)

        dense.append(x_)

    x = tf.keras.layers.Average()(dense)

    

    

    x = Dense(1, activation='sigmoid', name='fc3')(x)

    

    model = Model(inputs=input_img, outputs=x)

    return model
model =  build_model()

model.summary()
# Open the VGG16 weight file

#https://www.kaggle.com/aakashnain/beating-everything-with-depthwise-convolution

import h5py

f = h5py.File('../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')



# Select the layers for which you want to set weight.



w,b = f['block1_conv1']['block1_conv1_W_1:0'], f['block1_conv1']['block1_conv1_b_1:0']

model.layers[1].set_weights = [w,b]



w,b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']

model.layers[2].set_weights = [w,b]



w,b = f['block2_conv1']['block2_conv1_W_1:0'], f['block2_conv1']['block2_conv1_b_1:0']

model.layers[4].set_weights = [w,b]



w,b = f['block2_conv2']['block2_conv2_W_1:0'], f['block2_conv2']['block2_conv2_b_1:0']

model.layers[5].set_weights = [w,b]



f.close()

model.summary()  
model.compile(

            optimizer = CFG['optimizer'],

            loss      =  tf.keras.losses.BinaryCrossentropy(label_smoothing = CFG['label_smooth_fac']),

            metrics   = [tf.keras.metrics.AUC(name='auc')])
%%time

# from : https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords/comments?

skf = KFold(n_splits=2,shuffle=True,random_state=42)

oof_pred = []; oof_tar = []; oof_val = []; oof_names = [] 



GCS_PATH = [None]*2; IMG_SIZES = [256,256]



for i,k in enumerate(IMG_SIZES):

    GCS_PATH[i] = KaggleDatasets().get_gcs_path('melanoma-%ix%i'%(k,k))

    #GCS_PATH2[i] = KaggleDatasets().get_gcs_path('isic2019-%ix%i'%(k,k))

files_train = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/train*.tfrec')))

files_test  = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/test*.tfrec')))



for fold,(idxT,idxV) in enumerate(skf.split(np.arange(15))):

    

    # DISPLAY FOLD INFO

    if DEVICE=='TPU':

        if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)

    print('_'*25); print('**** FOLD',fold+1)

    # CREATE TRAIN AND VALIDATION SUBSETS

    files_train = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec'%x for x in idxT])

    files_valid = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec'%x for x in idxV])

    

    # SAVE BEST MODEL EACH FOLD

    sv = tf.keras.callbacks.ModelCheckpoint(

        'vgg16fold-%i.h5'%fold, monitor='val_loss', verbose=1, save_best_only=True,

        save_weights_only=True, mode='min', save_freq='epoch')

    

    steps_per_epoch  = count_data_items(files_train) / (CFG['batch_size'] * REPLICAS)



    history_Xception1   = model.fit(

        get_dataset(files_train, CFG,augment=True, shuffle=True, repeat=True), 

        epochs=40, callbacks = [sv,get_lr_callback(CFG)], 

        steps_per_epoch=count_data_items(files_train)/128//REPLICAS,

        validation_data=get_dataset(files_valid,CFG,augment=False,shuffle=False,

                repeat=False), 

        verbose=1

    )

    # PREDICT OOF USING TTA

    TTA = 2

    print('Predicting OOF with TTA...')

    ds_valid = get_dataset(files_valid,CFG,labeled=False,return_image_names=False,augment=True,

            repeat=True,shuffle=False,)

    ct_valid = count_data_items(files_valid); 

    STEPS = TTA * ct_valid/128/REPLICAS

    pred = model.predict(ds_valid,steps=STEPS,verbose=1)[:TTA*ct_valid,] 



    oof_pred.append( np.mean(pred.reshape((ct_valid,TTA),order='F'),axis=1) )                 



    # GET OOF TARGETS AND NAMES

    ds_valid = get_dataset(files_valid,CFG, augment=False, repeat=False,

            labeled=True, return_image_names=True)

    oof_tar.append( np.array([target.numpy() for img, target in iter(ds_valid.unbatch())]) )



    # REPORT RESULTS

    auc = roc_auc_score(oof_tar[-1],oof_pred[-1])

    oof_val.append(np.max( history_Xception1.history['val_auc'] ))

    print('#### FOLD %i OOF AUC with TTA = %.3f, without TTA = %.3f'%(fold+1,auc,oof_val[-1]))

    print()   

    

    

    for i in range(len(oof_pred[0])):

        if(oof_pred[0][i] > 0.5):

            oof_pred[0][i] = 1

        else:

            oof_pred[0][i] = 0

        

    oof_preds = []

    oof_tars = []

    for i in range(len(oof_pred[0])):

        oof_preds.append(oof_pred[0][i])

        oof_tars.append(oof_tar[0][i])





    # Get the confusion matrix

    cm  = confusion_matrix(oof_tars, oof_preds)

    plt.figure()

    plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True)

    plt.yticks(range(2), ['Normal', 'melanoma'], fontsize=16)

    plt.show()

    

    # Calculate Precision and Recall

    tn, fp, fn, tp = cm.ravel()



    precision = tp/(tp+fp)

    recall = tp/(tp+fn)



    print("Recall of the model is {:.2f}".format(recall))

    print("Precision of the model is {:.2f}".format(precision))



    
# COMPUTE OVERALL OOF AUC

oof = np.concatenate(oof_pred)

true = np.concatenate(oof_tar)

auc = roc_auc_score(true,oof)

print('Overall OOF AUC with TTA = %.3f'%auc)



# SAVE OOF TO DISK

df_oof = pd.DataFrame(dict( pred = oof, target=true))

df_oof.to_csv('oof.csv',index=False)

df_oof.head()
oof_tars = []



oof_preds = []



for i in range(len(df_oof)):

    if(df_oof.pred[i] > 0.5):

        oof_preds.append(1)

        oof_tars.append(df_oof.target[i])

    else:

        oof_preds.append(0)

        oof_tars.append(df_oof.target[i])





# Get the confusion matrix

cm  = confusion_matrix(oof_tars, oof_preds)

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True)

plt.yticks(range(2), ['Normal', 'melanoma'], fontsize=16)

plt.show()



# Calculate Precision and Recall

tn, fp, fn, tp = cm.ravel()



precision = tp/(tp+fp)

recall = tp/(tp+fn)



print("Overall Recall of the model is {:.2f}".format(recall))

print("Overall Precision of the model is {:.2f}".format(precision))
history_Xception1.history['lr']
%%time

CFG['batch_size'] = 256



cnt_test   = count_data_items(files_test)

steps      = cnt_test / (CFG['batch_size'] * REPLICAS) * CFG['tta_steps']

ds_testAug = get_dataset(files_test, CFG, augment=True, repeat=True, 

                         labeled=False, return_image_names=False)

preds = model.predict(ds_testAug, verbose=1, steps=steps)

print("Test shape :",df_sub.shape)

print("Preds shape :",preds.shape)
# https://www.kaggle.com/vbhargav875/efficientnet-b5-b6-b7-tf-keras



preds = preds[:,:cnt_test* CFG['tta_steps']]

preds = preds[:df_test.shape[0]*CFG['tta_steps']]

preds = np.stack(np.split(preds, CFG['tta_steps']),axis=1)

preds = np.mean(preds, axis=1)

preds = preds.reshape(-1)
print("New Preds shape :",preds.shape)
ds = get_dataset(files_test, CFG, augment=False, repeat=False, 

                 labeled=False, return_image_names=True)



image_names = np.array([img_name.numpy().decode("utf-8") 

                        for img, img_name in iter(ds.unbatch())])
submission = pd.DataFrame(dict(

        image_name = image_names,

        target = preds ))



submission = submission.sort_values('image_name') 

submission.to_csv(f'submission_vgg16.csv', index=False)
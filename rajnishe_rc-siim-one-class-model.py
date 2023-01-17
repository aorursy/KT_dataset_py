# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
faeture_list = ['image_name','target','tfrecord']



siim20_csv = pd.read_csv('../input/jpeg-melanoma-384x384/train.csv',usecols=faeture_list)

siim19_csv = pd.read_csv('../input/jpeg-isic2019-384x384/train.csv',usecols=faeture_list)
siim20_csv.info()
siim20_images = list(os.listdir('../input/jpeg-melanoma-384x384/train'))

siim19_images = list(os.listdir('../input/jpeg-isic2019-384x384/train'))
print ( 'ISIC_0071367.jpg' in siim19_images )
print(len(siim20_images))

print(len(siim19_images))
siim19_csv['year'] = '2019' 

siim20_csv['year'] = '2020'



siim_all = pd.concat([siim19_csv,siim20_csv],ignore_index = True)



train = siim_all.loc[siim_all.target == 0]
train.target.value_counts()
#SEED value

SEED_VALUE = 2244
#print(train.info())

#print(train.head(553))
filter_train = train[train.tfrecord != -1 ]
idx_list = []

for img_name in filter_train.image_name.values:

    if img_name.endswith('downsampled'):

        idx = filter_train.index[filter_train['image_name'] == img_name].to_list()

        #print(str(idx) + str(len(idx)) + ':' +img_name )

        if len(idx) == 1:

            idx_list.append(idx[0])
print(len(idx_list))

filter_train = filter_train.drop(idx_list)

# shuffle the rows

filter_train.reset_index(inplace=True)

filter_train.head()
filter_train.drop('index',axis=1)
cnt = 0

for img_name in filter_train.image_name.values:

#    if img_name == -1:

#        cnt = cnt + 1

#        print(train[])



    if img_name.endswith('downsampled'):

        cnt = cnt + 1

print(cnt)
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

import tensorflow_addons as tfa

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm



import efficientnet.tfkeras as efn



from kaggle_datasets import KaggleDatasets

from tensorflow.keras import backend as K

import tensorflow_addons as tfa
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
AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_PATH_19 = KaggleDatasets().get_gcs_path('jpeg-isic2019-384x384')

GCS_PATH_20 = KaggleDatasets().get_gcs_path('jpeg-melanoma-384x384')



# Configuration

EPOCHS = 3

BATCH_SIZE = 4 * strategy.num_replicas_in_sync

img_size = 384

IMAGE_SIZE = [img_size,img_size]
#print(filter_train.loc[filter_train.image_name == 'ISIC_0071367'].year.to_numpy()[0])
def add_gcs_path(image_id):

    

    year_nb = filter_train.loc[filter_train.image_name == image_id].year.to_numpy()[0]

    #print(year_nb)

    GCS_PATH = ''

    

    if year_nb == '2019':

        GCS_PATH = GCS_PATH_19 + '/train/' + image_id + '.jpg'

    else:

        GCS_PATH = GCS_PATH_20 + '/train/' + image_id + '.jpg'

    

    return GCS_PATH
def file_path(image_id):

    

    year_nb = filter_train.loc[filter_train.image_name == image_id].year.to_numpy()[0]

    #print(year_nb)

    GCS_PATH = ''

    

    if year_nb == '2019':

        #print('19')

        GCS_PATH = '../input/jpeg-isic2019-384x384' + '/train/' + image_id + '.jpg'

    else:

        #print('20')

        GCS_PATH = '../input/jpeg-melanoma-384x384' + '/train/' + image_id + '.jpg'

    

    return GCS_PATH
filter_train["image_path"] = filter_train["image_name"].apply(lambda x : add_gcs_path(x))

filter_train["image_jpg_id"] = filter_train["image_name"].apply(lambda x: file_path(x))
filter_train.head()
#filter_train[filter_train.image_name == 'ISIC_0071367']
xtrain, xval, ytrain, yval = train_test_split(filter_train["image_jpg_id"], filter_train["target"], 

                                              test_size = 0.10, stratify = filter_train["target"],

                                              random_state=SEED_VALUE)



df_train = pd.DataFrame({"image_path":xtrain, "target":ytrain})

df_val = pd.DataFrame({"image_path":xval, "target":yval})



df_train["target"] = df_train["target"].astype('int')

df_val["target"] = df_val["target"].astype('int')
#print(df_train.target.value_counts())

#print(df_val.target.value_counts())
train_paths = df_train.image_path.values

val_paths   = df_val.image_path.values



train_labels = df_train.target

val_labels   = df_val.target
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



def transform(image, label):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, sheared, zoomed, and shifted

    DIM = IMAGE_SIZE[0]

    XDIM = DIM%2 #fix for size 331

    

    if 0.5 > tf.random.uniform([1], minval = 0, maxval = 1):

        rot = 15. * tf.random.normal([1],dtype='float32')

    else:

        rot = 180. * tf.random.normal([1],dtype='float32')

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

        

    return image , label

    #return {'inp1': tf.reshape(d,[DIM,DIM,3]), 'inp2': image['inp2']}, label



def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)



def decode_image(filename, label=None, image_size=(img_size, img_size)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, size = image_size)

    

    if label is None:

        return image

    else:

        return image, label



def int_div_round_up(a, b):

    return (a + b - 1) // b
def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    image = tf.image.transpose(image)

    image = tf.image.rot90(image)

    image = tf.image.random_saturation(image, 0.7, 1.3)

    image = tf.image.random_contrast(image, 0.8, 1.2)

    image = tf.image.random_brightness(image, 0.1)    

    # used in Christ's notebook

    #image = tf.image.random_saturation(image, 0, 2)

    #imgage = tf.image.random_contrast(img, 0.8, 1.2)

    #imgage = tf.image.random_brightness(img, 0.1)



    return image, label
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .map(data_augment, num_parallel_calls=AUTO)

    #.map(transform, num_parallel_calls = AUTO)

    .repeat()

    .shuffle(512)

    .batch(BATCH_SIZE)

    .prefetch(AUTO))



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((val_paths, val_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

    .prefetch(AUTO))



NUM_TRAINING_IMAGES = df_train.shape[0]

NUM_VALIDATION_IMAGES = df_val.shape[0]



STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

VALIDATION_STEPS = int_div_round_up(NUM_VALIDATION_IMAGES, BATCH_SIZE)

print('Dataset: {} training images, {} validation images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES))

LR = .00003



def get_model():

    with strategy.scope():

        img_input = tf.keras.layers.Input(shape = (*IMAGE_SIZE, 3))

        

        base = efn.EfficientNetB4(weights = 'imagenet', include_top = False)

        #base = tf.keras.applications.ResNet152V2(weights = 'imagenet', include_top = False)

        

        x = base(img_input)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        #x = tf.keras.layers.Dropout(0.3)(x)

       #x = tf.keras.layers.Dense(128, activation = 'relu')(x)

    

        output = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

        

        model = tf.keras.models.Model(inputs = img_input, outputs = output)

        

        #opt = tf.keras.optimizers.Adam(learning_rate = LR)

        #tfa.losses.SigmoidFocalCrossEntropy(gamma = 2.0, alpha = 0.80)

        opt = tfa.optimizers.RectifiedAdam(lr=LR)

    

        model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = [ 'accuracy' ] )

    #tf.keras.metrics.AUC()

    return model
model_fold_1 = get_model()

print(model_fold_1.summary())
#lr scheduler

cb_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.4, 

                                                      patience = 2, verbose = 1, min_delta = 0.0001, mode = 'auto')
# ISIC_0071367

print ( 'ISIC_0071367.jpg' in siim19_images )

print ( 'ISIC_0071367.jpg' in siim20_images )
history1 = model_fold_1.fit(

    train_dataset, 

    epochs = EPOCHS, 

    callbacks = [cb_lr_schedule],

    steps_per_epoch = STEPS_PER_EPOCH,

    validation_data = valid_dataset,

    validation_steps = VALIDATION_STEPS

)
acc = history1.history['auc']

val_acc = history1.history['val_auc']



loss = history1.history['loss']

val_loss = history1.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'b', label='Training acc')

plt.plot(epochs, val_acc, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

 

plt.figure()

 

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
model_fold_1.save('B4-one-class-m0_leakyrelu.h5')
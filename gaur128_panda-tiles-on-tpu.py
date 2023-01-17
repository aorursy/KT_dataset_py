!pip install -q efficientnet
import efficientnet.tfkeras as efn
import math, re, os, time, gc
import skimage.io
import PIL
import time
import math
import warnings
import cv2
import numpy as np
import pandas as pd
from collections import namedtuple
import tensorflow as tf
import albumentations as albu
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from kaggle_datasets import KaggleDatasets
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import cohen_kappa_score, make_scorer

SEED = 2020
warnings.filterwarnings('ignore')
print('Tensorflow version : {}'.format(tf.__version__))
AUTO = tf.data.experimental.AUTOTUNE
# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)
MIXED_PRECISION = False
XLA_ACCELERATE = False

if MIXED_PRECISION:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    if tpu: policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    else: policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Mixed precision enabled')

if XLA_ACCELERATE:
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')
!ls /kaggle/input
MAIN_DIR = '../input/prostate-cancer-grade-assessment'
TRAIN_IMG_DIR = '../input/panda-2020-level-1-2/train_images/train_images'
TEST_IMG_DIR = '../input/prostate-cancer-grade-assessment/test_images'
SAMPLE = '../input/prostate-cancer-grade-assessment/sample_submission.csv'
sub_df  = pd.read_csv(os.path.join(MAIN_DIR, 'sample_submission.csv'))
test_df = pd.read_csv(os.path.join(MAIN_DIR, 'test.csv')).set_index('image_id')
df = pd.read_csv(os.path.join(MAIN_DIR, 'train.csv')).set_index('image_id')

files = sorted(set([p[:32] for p in os.listdir(TRAIN_IMG_DIR)]))
df = df.loc[files]
train_csv = df.reset_index()

# Wrongly labeled data
wrong_label = train_csv[(train_csv['isup_grade'] == 2) & (train_csv['gleason_score'] == '4+3')]
print(wrong_label)
train_csv.drop([wrong_label.index[0]],inplace=True)
train_csv = train_csv.reset_index()

# incosistency with "0" and "negative"
train_csv['gleason_score'] = train_csv['gleason_score'].apply(lambda x: "0+0" if x=="negative" else x)
radboud_csv = train_csv[train_csv['data_provider'] == 'radboud']
karolinska_csv = train_csv[train_csv['data_provider'] != 'radboud']

# GCS_DS_PATH = KaggleDatasets().get_gcs_path('prostate-cancer-grade-assessment')
# GCS_DS_PATH = KaggleDatasets().get_gcs_path('panda-15x256x256-tiles-merged')
# GCS_DS_PATH = KaggleDatasets().get_gcs_path('panda-4x512x512-merged-tiles')
# GCS_DS_PATH = 'gs://kds-10bdb17716a14253285e22f42916ffb63fb2b65e918871e28846bed6' # For panda-15x256x256-tiles-merged
# GCS_DS_PATH = 'gs://kds-eecf7646d11ffc661be4348df45875cb14d6dddf516071572ea7cc12' # For panda-4x512x512-merged-tiles
# GCS_DS_PATH = 'gs://kds-ada7d35cbdd41095c77a3724f32409454cb4258b266b089616a4cc5a'
# GCS_DS_PATH = KaggleDatasets().get_gcs_path('panda-2020-level-1-2')
GCS_DS_PATH = 'gs://kds-cc0d0c404acc1bbeeb5e049026f19daad1ba110ea6a6ea26434c78af'
GCS_DS_PATH
# !gsutil ls $GCS_PATH
splits = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
splits = list(splits.split(radboud_csv, radboud_csv.isup_grade))
fold_splits = np.zeros(len(radboud_csv)).astype(np.int)
for i in range(5): 
    fold_splits[splits[i][1]]=i
radboud_csv['fold'] = fold_splits
radboud_csv.head(5)

splits = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
splits = list(splits.split(karolinska_csv, karolinska_csv.isup_grade))
fold_splits = np.zeros(len(karolinska_csv)).astype(np.int)
for i in range(5): 
    fold_splits[splits[i][1]]=i
karolinska_csv['fold'] = fold_splits
karolinska_csv.head(5)

train_csv = pd.concat([radboud_csv, karolinska_csv])
train_csv.shape
TRAIN_FOLD = 0
train_df = train_csv[train_csv['fold'] != TRAIN_FOLD]
valid_df = train_csv[train_csv['fold'] == TRAIN_FOLD]

print(train_df.shape)
print(valid_df.shape)
train_paths = train_df["image_id"].apply(lambda x: GCS_DS_PATH + '/train_images/train_images/' + x + '_2.jpeg').values
valid_paths = valid_df["image_id"].apply(lambda x: GCS_DS_PATH + '/train_images/train_images/' + x + '_2.jpeg').values
train_labels = pd.get_dummies(train_df['isup_grade']).astype('int32').values
valid_labels = pd.get_dummies(valid_df['isup_grade']).astype('int32').values

print(train_labels.shape) 
print(valid_labels.shape)
train_labels[1]
%%time

# Further target pre-processing

# Instead of predicting a single label, we will change our target to be a multilabel problem; 
# i.e., if the target is a certain class, then it encompasses all the classes before it. 
# E.g. encoding a class 4 retinopathy would usually be [0, 0, 0, 1], 
# but in our case we will predict [1, 1, 1, 1]. For more details, 
# please check out Lex's kernel.

train_labels_multi = np.empty(train_labels.shape, dtype=train_labels.dtype)
train_labels_multi[:, 5] = train_labels[:, 5]

for i in range(4, -1, -1):
    train_labels_multi[:, i] = np.logical_or(train_labels[:, i], train_labels_multi[:, i+1])

print("Original y_train:", train_labels.sum(axis=0))
print("Multilabel version:", train_labels_multi.sum(axis=0))
%%time

# Further target pre-processing

# Instead of predicting a single label, we will change our target to be a multilabel problem; 
# i.e., if the target is a certain class, then it encompasses all the classes before it. 
# E.g. encoding a class 4 retinopathy would usually be [0, 0, 0, 1], 
# but in our case we will predict [1, 1, 1, 1]. For more details, 
# please check out Lex's kernel.

valid_labels_multi = np.empty(valid_labels.shape, dtype=valid_labels.dtype)
valid_labels_multi[:, 5] = valid_labels[:, 5]

for i in range(4, -1, -1):
    valid_labels_multi[:, i] = np.logical_or(valid_labels[:, i], valid_labels_multi[:, i+1])

print("Original y_train:", valid_labels.sum(axis=0))
print("Multilabel version:", valid_labels_multi.sum(axis=0))
# train_paths1 = train_df["image_id"].apply(lambda x: TRAIN_IMG_DIR + '/' + x + '_2.jpeg').values
# valid_paths1 = valid_df["image_id"].apply(lambda x: TRAIN_IMG_DIR + '/' + x + '_2.jpeg').values
# X_MAX = 0
# Y_MAX = 0
# for i in range(len(train_paths1)):
#     img = cv2.imread(train_paths1[i])
#     if (X_MAX<img.shape[0]):
#         X_MAX = img.shape[0]
#     if (Y_MAX<img.shape[1]):
#         Y_MAX = img.shape[1]

        
# for i in range(len(valid_paths1)):
#     img = cv2.imread(valid_paths1[i])
#     if (X_MAX<img.shape[0]):
#         X_MAX = img.shape[0]
#     if (Y_MAX<img.shape[1]):
#         Y_MAX = img.shape[1]

# print(X_MAX)
# print(Y_MAX)
# For Medium resolution
# X_MAX = 11768
# Y_MAX = 24152
# For Low resolution
X_MAX = 2944
Y_MAX = 6040
IMAGE_SIZE = (256, 256) # at this size, a GPU will run out of memory. Use the TPU
EPOCHS = 15
BATCH_SIZE = 64 * strategy.num_replicas_in_sync
FOLDS = 1
SEED = 777
CHANNELS = 3
HEIGHT = IMAGE_SIZE[0]
WIDTH = IMAGE_SIZE[1]
VALIDATION = True
CLASSES = 6
N = 36
N_2 = 6
SZ = 128
MODELNAME = 'EfficientNetB3'
# For Low Images 36x128x128 are enough
# For Medium Images 64x256x256/36x512x512 are enough
FOLDED_NUM_TRAIN_IMAGES = train_df.shape[0]
FOLDED_NUM_VALID_IMAGES = valid_df.shape[0]
STEPS_PER_EPOCH = FOLDED_NUM_TRAIN_IMAGES // BATCH_SIZE
VALIDATION_STEPS = FOLDED_NUM_VALID_IMAGES // BATCH_SIZE
print('*'*20)
print('Notebook info')
print('Training data : {}'.format(FOLDED_NUM_TRAIN_IMAGES))
print('Validing data : {}'.format(FOLDED_NUM_VALID_IMAGES))
print('Categorical classes : {}'.format(CLASSES))
print('Training image size : {}'.format(IMAGE_SIZE))
print('Training epochs : {}'.format(EPOCHS))
print('*'*20)
# numpy and matplotlib defaults
np.set_printoptions(threshold=15, linewidth=80)

def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images, numpy_labels

def display_one_image(image, title, subplot, red=False, titlesize=16):
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)
    
def display_batch_of_images(databatch, predictions=None):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
   
    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images)//rows
        
    # size and spacing
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols,1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))
    
    # display
    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
        title = label
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_image(image, title, subplot, not correct, titlesize=dynamic_titlesize)
    
    #layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()
    
def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])
def augment_images(images):
    images = tf.image.transpose(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.random_flip_left_right(images)
    return images   

# @tf.function
def decode_image_n(filename, label=None):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.int32) - 255
    image = tf.image.resize_with_pad(image, X_MAX, Y_MAX, method=tf.image.ResizeMethod.BILINEAR, antialias=False)
    image = tf.expand_dims(image, 0)

    tiles = tf.image.extract_patches(image, sizes = [1, SZ, SZ, 1], strides = [1, SZ, SZ, 1], rates=[1, 1, 1, 1], padding = 'VALID')
    tiles = tf.reshape(tiles, [-1, SZ, SZ, 3])
    
    sort = tf.math.reduce_sum(tf.reshape(tiles, [tiles.shape[0], -1]), axis = 1)
    sort = tf.argsort(sort, axis=0, direction='ASCENDING', stable=False, name=None)

    tiles = tf.gather(tiles, sort[:N])
    tiles = augment_images(tiles)

    H_CONCAT = []
    index = 0
    for j in range(N_2):
        V_CONCAT = []
        for i in range(N_2):
            V_CONCAT.append(tiles[index])
            index+=1
        H_CONCAT.append(tf.concat(V_CONCAT, axis = 0))
        
    tiles = tf.concat(H_CONCAT, axis = 1)
    image = tf.cast(tiles, tf.int32) + 255
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, IMAGE_SIZE)

    if label is None:
        return image
    else:
        return image, label
def decode_image(filename, label=None):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, IMAGE_SIZE)
    if label is None:
        return image
    else:   
        return image, label
def load_dataset(filenames, filelabels, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels)) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
#     dataset = dataset.map(decode_image, num_parallel_calls=AUTO)
    dataset = dataset.map(decode_image_n, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def get_training_dataset(dataset = None, do_aug=True, file_names = None, file_labels = None):
    if dataset == None: dataset = load_dataset(filenames = file_names, filelabels = file_labels, labeled=True)
    dataset = dataset.shuffle(count_data_items(file_names))
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    
    if do_aug:
        dataset = dataset.map(data_augment_basic, num_parallel_calls=AUTO)
#         dataset = dataset.map(transform_rotate, num_parallel_calls=AUTO)
    
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(dataset = None, ordered=False, file_names = None, file_labels = None):
    if dataset == None: dataset = load_dataset(filenames = file_names, filelabels = file_labels, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    return len(filenames)
def data_augment_basic(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.transpose(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, 0, 2)
    return image, label   

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
def transform_rotate(image, label):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = IMAGE_SIZE[0]
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
    
    return tf.reshape(d,[DIM,DIM,3]), label
# # data dump
# print("Training data shapes:")
# for image, label in get_training_dataset(do_aug=False, file_names=TRAINING_FILENAMES).take(3):
#     print(image.numpy().shape, label.numpy().shape)
# print("Training data label examples:", label.numpy())
# print("Validation data shapes:")
# for image, label in get_validation_dataset().take(3):
#     print(image.numpy().shape, label.numpy().shape)
# print("Validation data label examples:", label.numpy())
# print("Test data shapes:")
# for image, idnum in get_test_dataset().take(3):
#     print(image.numpy().shape, idnum.numpy().shape)
# print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string
# Peek at training data
training_dataset = get_training_dataset(do_aug=False, file_names = train_paths[:2], file_labels = train_labels[:2])
training_dataset = training_dataset.unbatch().batch(2)
train_batch = iter(training_dataset)
# run this cell again for next set of images
display_batch_of_images(next(train_batch))
# # peer at test data
# test_dataset = get_test_dataset()
# test_dataset = test_dataset.unbatch().batch(20)
# test_batch = iter(test_dataset)
# # run this cell again for next set of images
# display_batch_of_images(next(test_batch))
# # Learning rate schedule for TPU, GPU and CPU.
# # Using an LR ramp up because fine-tuning a pre-trained model.
# # Starting with a high LR would break the pre-trained weights.

# LR_START = 0.00001
# LR_MAX = 0.001 * strategy.num_replicas_in_sync
# LR_MIN = 0.00001
# LR_RAMPUP_EPOCHS = 3
# LR_SUSTAIN_EPOCHS = 1
# LR_EXP_DECAY = .8

# # in custom training loop training you need an object to hold the epoch value
# class LRSchedule():
#     def __init__(self):
#         self.epoch = 0
        
#     def set_epoch(self, epoch):
#         self.epoch = epoch
        
#     @staticmethod
#     def lrfn(epoch):
#         if epoch < LR_RAMPUP_EPOCHS:
#             lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
#         elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
#             lr = LR_MAX
#         else:
#             lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
#         return lr
    
#     def lr(self):
#         return self.lrfn(self.epoch)
    
#     # LR scaled by 8 for CTL
#     # not quite sure yet why LR must be scaled up by 8 (otherwise, does not converge the same)
#     def lr_scaled(self):
#         return self.lrfn(self.epoch) * 8
    

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(LRSchedule.lrfn, verbose=True)

# rng = [i for i in range(EPOCHS)]
# y = [LRSchedule.lrfn(x) for x in rng]
# plt.plot(rng, y)
# print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
# def get_model():
#     with strategy.scope():
#         pretrained_model = efn.EfficientNetB3(weights='noisy-student', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
# #         pretrained_model = tf.keras.applications.ResNet152V2(input_shape=[*IMAGE_SIZE, 3], include_top=False, weights='imagenet')
#         pretrained_model.trainable = True # transfer learning
        
#         model = tf.keras.Sequential([
#             pretrained_model,
#             tf.keras.layers.GlobalAveragePooling2D(),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Dropout(0.25),
#             tf.keras.layers.Dense(512, activation='elu'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Dropout(0.25),
#             tf.keras.layers.Dense(CLASSES, activation='softmax')
#         ])
        
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(lr=1e-3),
#             loss = 'categorical_crossentropy',
#             metrics=['categorical_accuracy']
#         )

#     return model

LR_START = 0.00001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = .8
        
@tf.function
def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
with strategy.scope():
    pretrained_model = efn.EfficientNetB3(weights='noisy-student', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#         pretrained_model = tf.keras.applications.ResNet152V2(input_shape=[*IMAGE_SIZE, 3], include_top=False, weights='imagenet')
    pretrained_model.trainable = True # transfer learning
        
    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(512, activation='elu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(CLASSES, activation='softmax')
    ])
    
    # Instiate optimizer with learning rate schedule
    class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            return lrfn(epoch=step//STEPS_PER_EPOCH)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LRSchedule())

    train_loss = tf.keras.metrics.Sum()
    valid_loss = tf.keras.metrics.Sum()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    valid_accuracy = tf.keras.metrics.CategoricalAccuracy()
    
#     loss_fn = tf.keras.losses.categorical_crossentropy
    loss_fn = tf.keras.losses.MSLE
# model_file = '{}/model_{}.h5'.format('.', 'V0')
# earlystopper = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', 
#     patience=5, 
#     verbose=1,
#     mode='min'
# )
# modelsaver = tf.keras.callbacks.ModelCheckpoint(
#     model_file, 
#     monitor='val_loss', 
#     verbose=1, 
#     save_best_only=True,
#     mode='min'
# )
# lrreducer = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=.1,
#     patience=3,
#     verbose=1,
#     min_lr=1e-7
# )
@tf.function
def qw_kappa_score(y_true, y_pred):     
#     y_true=tf.math.argmax(y_true, axis=1)
#     y_pred=tf.math.argmax(y_pred, axis=1)
    def sklearn_qwk(y_true, y_pred) -> np.float64:
        return cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return tf.compat.v1.py_func(sklearn_qwk, (y_true, y_pred), tf.double)
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        probabilities = model(images, training=True)
        loss = loss_fn(labels, probabilities)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # update metrics
    train_accuracy.update_state(labels, probabilities)
    train_loss.update_state(loss)

@tf.function
def valid_step(images, labels):
    probabilities = model(images, training=False)
    loss = loss_fn(labels, probabilities)
    
    # update metrics
    valid_accuracy.update_state(labels, probabilities)
    valid_loss.update_state(loss)
    
    return probabilities, labels
start_time = epoch_start_time = time.time()

# distribute the datset according to the strategy
train_dist_ds = strategy.experimental_distribute_dataset(get_training_dataset(do_aug=True, file_names = train_paths, file_labels = train_labels_multi))
valid_dist_ds = strategy.experimental_distribute_dataset(get_validation_dataset(file_names = valid_paths, file_labels = valid_labels_multi))

print("Steps per epoch:", STEPS_PER_EPOCH)
History = namedtuple('History', 'history')
history = History(history={'loss': [], 'val_loss': [], 'categorical_accuracy': [], 'val_categorical_accuracy': [], 'qwk':[]})

epoch = 0
for step, (images, labels) in enumerate(train_dist_ds):
    
    # run training step
    strategy.experimental_run_v2(train_step, args=(images, labels))
    print('=', end='', flush=True)

    # validation run at the end of each epoch
    if ((step+1) // STEPS_PER_EPOCH) > epoch:
        print('|', end='', flush=True)
        
        # validation run
        for image, labels in valid_dist_ds:
            probabilities, labels = strategy.experimental_run_v2(valid_step, args=(image, labels))
            print('=', end='', flush=True)

        # compute metrics
#         qwk = qw_kappa_score(groundtruths, prediction)
        history.history['categorical_accuracy'].append(train_accuracy.result().numpy())
        history.history['val_categorical_accuracy'].append(valid_accuracy.result().numpy())
        history.history['loss'].append(train_loss.result().numpy() / STEPS_PER_EPOCH)
        history.history['val_loss'].append(valid_loss.result().numpy() / VALIDATION_STEPS)
#         history.history['qwk'].append(qwk)
        
        # report metrics
        epoch_time = time.time() - epoch_start_time
        print('\nEPOCH {:d}/{:d}'.format(epoch+1, EPOCHS))
        print('time: {:0.1f}s'.format(epoch_time),
              'loss: {:0.4f}'.format(history.history['loss'][-1]),
              'accuracy: {:0.4f}'.format(history.history['categorical_accuracy'][-1]),
              'val_loss: {:0.4f}'.format(history.history['val_loss'][-1]),
              'val_acc: {:0.4f}'.format(history.history['val_categorical_accuracy'][-1]),
              'lr: {:0.4g}'.format(lrfn(epoch)), flush=True)
        
        # set up next epoch
        epoch = (step+1) // STEPS_PER_EPOCH
        epoch_start_time = time.time()
        train_accuracy.reset_states()
        valid_accuracy.reset_states()
        valid_loss.reset_states()
        train_loss.reset_states()
        
        if epoch >= EPOCHS:
            break

simple_ctl_training_time = time.time() - start_time
print("SIMPLE CTL TRAINING TIME: {:0.1f}s".format(simple_ctl_training_time))
model.save(MODELNAME + "_model.h5")
display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)
display_training_curves(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'], 'accuracy', 212)
# def train():
        
#     model = get_model()
#     history = model.fit(
#         get_training_dataset(do_aug=True, file_names = train_paths, file_labels = train_labels_multi), 
#         steps_per_epoch=STEPS_PER_EPOCH,
#         epochs=EPOCHS, 
#         validation_data=get_validation_dataset(file_names = valid_paths, file_labels = valid_labels_mul),
#         validation_steps=VALIDATION_STEPS,
#         callbacks=[lrreducer,modelsaver,earlystopper]
#     )
#     model.save(MODELNAME + "_model.h5")
#     return history, model
# start_time = time.time()

# # run train and predict
# history, model = train()
# display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)
# display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)

# keras_fit_training_time = time.time() - start_time
# print("KERAS FIT TRAINING TIME: {:0.1f}s".format(keras_fit_training_time))
# cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.
# images_ds = cmdataset.map(lambda image, label: image)
# labels_ds = cmdataset.map(lambda image, label: label).unbatch()
# cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch
# cm_probabilities = model.predict(images_ds)
# cm_predictions = np.argmax(cm_probabilities, axis=-1)
# print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)
# print("Predicted labels: ", cm_predictions.shape, cm_predictions)
# cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))
# score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
# precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
# recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
# #cmat = (cmat.T / cmat.sum(axis=1)).T # normalized
# display_confusion_matrix(cmat, score, precision, recall)
# print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
dataset = get_validation_dataset(file_names = valid_paths, file_labels = valid_labels_multi)
dataset = dataset.unbatch().batch(20)
batch = iter(dataset)
# # run this cell again for next set of images
images, labels = next(batch)
print(labels)
probabilities = model.predict(images)
print(probabilities)
predictions = np.argmax(probabilities, axis=-1)
print(predictions)
# display_batch_of_images((images, labels), predictions)
preds = probabilities > 0.37757874193797547
preds = preds.astype(int).sum(axis=1)
preds
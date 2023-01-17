import numpy as np
import pandas as pd
import os
import random, re, math
import tensorflow as tf, tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from kaggle_datasets import KaggleDatasets
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as L
from tensorflow.keras.applications import ResNet152V2, InceptionResNetV2, InceptionV3, Xception, VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint , LearningRateScheduler
from keras import regularizers

import matplotlib.pyplot as plt

!pip install efficientnet
import efficientnet.tfkeras as efn
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

GCS_DS_PATH_TEST = KaggleDatasets().get_gcs_path('plant-pathology-2020-fgvc7')
GCS_DS_PATH_TRAIN = KaggleDatasets().get_gcs_path('plant-pathology-2020-fgvc7')
path='../input/plant-pathology-2020-fgvc7/'
train_path = '../input/plant-pathology-2020-fgvc7/'
train = pd.read_csv(train_path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
sub = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')

train_paths = train.image_id.apply(lambda x: GCS_DS_PATH_TRAIN + '/images/' + x + '.jpg').values
test_paths = test.image_id.apply(lambda x: GCS_DS_PATH_TEST + '/images/' + x + '.jpg').values
train_labels = train.loc[:, 'healthy':].values
#from PIL import Image
#from numpy import asarray
#import numpy as np
#import cv2
#import os
#
#dataset = os.listdir('../input/plant-pathology-2020-fgvc7/images')
#
#pixel_num = 0 
#channel_sum = np.zeros(3)
#channel_sum_squared = np.zeros(3)
#
#for img in dataset:
#    image_path = os.path.join('../input/plant-pathology-2020-fgvc7/images', img)
#    im = cv2.imread(image_path) # image in M*N*CHANNEL_NUM shape, channel in BGR order
#    im = im/255.0
#    pixel_num += (im.size/3)
#    channel_sum += np.sum(im, axis=(0, 1))
#    channel_sum_squared += np.sum(np.square(im), axis=(0, 1))
#
#rgb_mean = channel_sum / pixel_num
#rgb_std = np.sqrt(channel_sum_squared / pixel_num - np.square(rgb_mean))
#
#print(rgb_mean)
#print(rgb_std)
#Get images mean and Std
mean = tf.constant([0.3230078  ,0.51456824 ,0.40597249], dtype=tf.float32)
tf.dtypes.cast(mean, tf.int32)
std = tf.constant([0.19176222 ,0.19101869 ,0.20486984], dtype=tf.float32)
tf.dtypes.cast(std, tf.int32)
#Calculate class weights
train.pop('image_id')
y_train = train.to_numpy().astype('float32')

category_names = ['healthy','multiple_diseases','rust','scab']
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced',np.unique(y_train.argmax(axis=1)),y_train.argmax(axis=1))
print('class weights: ',class_weights)

nb_classes = 4
BATCH_SIZE = 8
img_size = 512
EPOCHS = 20
SEED = 265
def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = ((tf.cast(image, tf.float32) / 255.0) - mean) / std
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label
    
def data_augment(image, label=None, seed=SEED):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
           
    if label is None:
        return image
    else:
        return image, label
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )

    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    
    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    
    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )
    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))
def transform(image,label=None):
    DIM = img_size
    XDIM = DIM%2 
    
    rot = 15. * tf.random.normal([1],dtype='float32')
    shr = 5. * tf.random.normal([1],dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.
    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.
    h_shift = 8. * tf.random.normal([1],dtype='float32') 
    w_shift = 8. * tf.random.normal([1],dtype='float32') 
  
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)
              
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image,tf.transpose(idx3))
    
    if label is None:
        return tf.reshape(d,[DIM,DIM,3])
    else:
        return tf.reshape(d,[DIM,DIM,3]),label
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .map(transform,num_parallel_calls=AUTO)
    .repeat()
    .shuffle(SEED)
    .batch(BATCH_SIZE)
   # .map(mixup)
    .prefetch(AUTO)
)
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)
LR_START = 0.0003
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 4
LR_SUSTAIN_EPOCHS = 2
LR_EXP_DECAY = .8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
with strategy.scope():
    enet = efn.EfficientNetB6(input_shape=(img_size, img_size, 3),weights='noisy-student',include_top=False)
    model2 = tf.keras.Sequential([enet,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(nb_classes,kernel_regularizer=regularizers.l2(0.02), activation='softmax')]) 
    model2.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003),loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02,reduction=tf.keras.losses.Reduction.NONE),metrics=['categorical_accuracy'])
#model2.load_weights('../input/plant-metalearning/B6_best_weights.h5')

with strategy.scope():
    enet = efn.EfficientNetB7(input_shape=(img_size, img_size, 3),weights='noisy-student',include_top=False)
    model3 = tf.keras.Sequential([enet,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(nb_classes,kernel_regularizer=regularizers.l2(0.02), activation='softmax')]) 
    model3.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003),loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02,reduction=tf.keras.losses.Reduction.NONE),metrics=['categorical_accuracy'])
#model3.load_weights('../input/plant-metalearning/B7_best_weights.h5')

with strategy.scope():
    enet = efn.EfficientNetB3(input_shape=(img_size, img_size, 3),weights='noisy-student',include_top=False)
    model4 = tf.keras.Sequential([enet,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(nb_classes,kernel_regularizer=regularizers.l2(0.02), activation='softmax')]) 
    model4.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003),loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02,reduction=tf.keras.losses.Reduction.NONE),metrics=['categorical_accuracy'])
#model4.load_weights('../input/plant-metalearning/B3_best_weights.h5')

with strategy.scope():
    enet = efn.EfficientNetB5(input_shape=(img_size, img_size, 3),weights='noisy-student',include_top=False)
    model5 = tf.keras.Sequential([enet,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(nb_classes,kernel_regularizer=regularizers.l2(0.02), activation='softmax')]) 
    model5.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003),loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02,reduction=tf.keras.losses.Reduction.NONE),metrics=['categorical_accuracy'])
#model5.load_weights('../input/plant-metalearning/B5_best_weights.h5')

with strategy.scope():
    enet = efn.EfficientNetB4(input_shape=(img_size, img_size, 3),weights='noisy-student',include_top=False)
    model6 = tf.keras.Sequential([enet,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(nb_classes,kernel_regularizer=regularizers.l2(0.02), activation='softmax')]) 
    model6.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003),loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02,reduction=tf.keras.losses.Reduction.NONE),metrics=['categorical_accuracy'])
#model6.load_weights('../input/plant-metalearning/B4_best_weights.h5')
model2.fit(
    train_dataset, 
    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
    callbacks=[lr_callback],
    epochs=EPOCHS,
   class_weight = class_weights)
model2.save_weights("B6_best_weights.h5")
model3.fit(
    train_dataset, 
   steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
   callbacks=[lr_callback],
   epochs=EPOCHS,
    class_weight = class_weights
)
model3.save_weights("B7_best_weights.h5")
model4.fit(
    train_dataset, 
    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
    callbacks=[lr_callback],
    epochs=EPOCHS,
    class_weight = class_weights
)
model4.save_weights("B3_best_weights.h5")
model5.fit(
    train_dataset, 
    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
    callbacks=[lr_callback],
    epochs=EPOCHS,
    class_weight = class_weights
)
model5.save_weights("B5_best_weights.h5")
model6.fit(
    train_dataset, 
    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
    callbacks=[lr_callback],
    epochs=EPOCHS,
    class_weight = class_weights
)
model6.save_weights("B4_best_weights.h5")
model33 = tf.keras.Sequential()
for layer in model3.layers[:-2]:
    model33.add(layer)
for layer in model33.layers:
    layer.trainable = False
    
model22 = tf.keras.Sequential()
for layer in model2.layers[:-2]:
    model22.add(layer)
for layer in model22.layers:
    layer.trainable = False
    
model44 = tf.keras.Sequential()
for layer in model4.layers[:-2]:
    model44.add(layer)
for layer in model44.layers:
    layer.trainable = False
    
#model55 = tf.keras.Sequential()
#for layer in model5.layers[:-2]:
#    model55.add(layer)
#for layer in model55.layers:
#    layer.trainable = False
    
model66 = tf.keras.Sequential()
for layer in model6.layers[:-2]:
    model66.add(layer)
for layer in model66.layers:
    layer.trainable = False
    
with strategy.scope():
    x = tf.keras.Input(shape = (img_size, img_size, 3))
    x2 = model22(x)
    x3 = model33(x)
    x4 = model44(x)
    # x55 = model55(x)
    x66 = model66(x)

    x5 = tf.keras.layers.concatenate([x2, x3, x4,x66], axis = 3) # x55, 
    x6 = tf.keras.layers.GlobalAveragePooling2D()(x5)
    #x6 = tf.keras.layers.Dropout(0.3)(x6)
    x7 = tf.keras.layers.Dense(1024,kernel_regularizer=regularizers.l2(0.02), activation='swish')(x6)
    x7 = tf.keras.layers.Dropout(0.3)(x7)
    x7 = tf.keras.layers.Dense(nb_classes, activation='softmax')(x7)
    out = tf.keras.Model(inputs = x, outputs = x7)

    out.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0003),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02,reduction=tf.keras.losses.Reduction.NONE),metrics=['categorical_accuracy'])


tf.keras.utils.plot_model(out, 'mini_out.png', show_shapes=True)
EPOCHS = 60
history = out.fit(
    train_dataset, 
    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
    callbacks=[lr_callback],
    epochs=12,
    class_weight = class_weights
)
out.save_weights("out.h5")
probs = out.predict(test_dataset, verbose=1)
sub = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')
sub.loc[:,'healthy':] = probs
sub.to_csv('final_meta_100_epoohs.csv', index = False)
probs2 = model2.predict(test_dataset, verbose=1)
probs3 = model3.predict(test_dataset, verbose=1)
probs4 = model4.predict(test_dataset, verbose=1)
probs5 = model5.predict(test_dataset, verbose=1)
probs6 = model6.predict(test_dataset, verbose=1)
probs_avg1 = 0.1*probs2 + 0.1*probs3 + 0.1*probs4+ 0.1*probs5 + 0.2*probs6 + 0.4*probs
sub2 = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')
sub2.loc[:, 'healthy':] = probs_avg1
sub2.to_csv('submission_ensemAvg_100_epoohs.csv', index=False)
sub2.head()
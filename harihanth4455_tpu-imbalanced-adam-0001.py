import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openslide
import os
import tensorflow as tf



from random import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from kaggle_datasets import KaggleDatasets
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from tqdm import tqdm
%matplotlib inline

print(tf.__version__)
print(tf.keras.__version__)
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



GCS_DS_PATH = KaggleDatasets().get_gcs_path('diabetic-retinopathy-resized')
train_df = pd.read_csv('/kaggle/input/diabetic-retinopathy-resized/trainLabels.csv')
print(train_df.shape)
from sklearn.model_selection import train_test_split
train,test = train_test_split(train_df,test_size = 0.16,random_state=1,stratify = train_df['level'])
print(train.shape) 
print(test.shape)
train,valid = train_test_split(train,test_size = 0.168,random_state=1,stratify = train['level'])
train_paths = train["image"].apply(lambda x: GCS_DS_PATH + '/resized_train/resized_train/' + x + '.jpeg').values
valid_paths = valid["image"].apply(lambda x: GCS_DS_PATH + '/resized_train/resized_train/' + x + '.jpeg').values
train_labels = pd.get_dummies(train['level']).astype('int32').values
valid_labels = pd.get_dummies(valid['level']).astype('int32').values

print(train_labels.shape) 
print(valid_labels.shape)
BATCH_SIZE= 8 * strategy.num_replicas_in_sync
img_size = 512
EPOCHS = 50
nb_classes = 5


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .repeat()
    .cache()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    )
valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)
from keras.optimizers import *
import keras
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
def get_resnet():
    with strategy.scope():
        resnet = keras.applications.resnet50.ResNet50(include_top = False, weights = 'imagenet', input_tensor = None, input_shape = (512,512,3))
        x = resnet.output
        x = Flatten()(x)
        output_layer = Dense(5,activation = 'softmax',name = 'softmax')(x)
        final_model = keras.Model(inputs = resnet.input, outputs = output_layer)
        opt = keras.optimizers.Nadam(lr = 0.0001,beta_1=0.9,beta_2=0.9)
        final_model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
        return final_model

def get_xception():
    with strategy.scope():
        resnet = keras.applications.Xception(include_top = False, weights = 'imagenet', input_tensor = None, input_shape = (512,512,3))
        x = resnet.output
        x = Flatten()(x)
        output_layer = Dense(5,activation = 'softmax',name = 'softmax')(x)
        final_model = keras.Model(inputs = resnet.input, outputs = output_layer)
        opt = keras.optimizers.Nadam(lr = 0.0001,beta_1=0.9,beta_2=0.9)
        final_model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
        return final_model
    
def get_inception():
    with strategy.scope():
        resnet = keras.applications.InceptionV3(include_top = False, weights = 'imagenet', input_tensor = None, input_shape = (512,512,3))
        x = resnet.output
        x = Flatten()(x)
        output_layer = Dense(5,activation = 'softmax',name = 'softmax')(x)
        final_model = keras.Model(inputs = resnet.input, outputs = output_layer)
        opt = keras.optimizers.Nadam(lr = 0.0001,beta_1=0.9,beta_2=0.9)
        final_model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
        return final_model
    
def get_dense121():
    with strategy.scope():
        resnet = keras.applications.DenseNet121(include_top = False, weights = 'imagenet', input_tensor = None, input_shape = (512,512,3))
        x = resnet.output
        x = Flatten()(x)
        output_layer = Dense(5,activation = 'softmax',name = 'softmax')(x)
        final_model = keras.Model(inputs = resnet.input, outputs = output_layer)
        opt = keras.optimizers.Nadam(lr = 0.0001,beta_1=0.9,beta_2=0.9)
        final_model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
        return final_model
    
def get_dense169():
    with strategy.scope():
        resnet = keras.applications.DenseNet169(include_top = False, weights = 'imagenet', input_tensor = None, input_shape = (512,512,3))
        x = resnet.output
        x = Flatten()(x)
        output_layer = Dense(5,activation = 'softmax',name = 'softmax')(x)
        final_model = keras.Model(inputs = resnet.input, outputs = output_layer)
        opt = keras.optimizers.Nadam(lr = 0.0001,beta_1=0.9,beta_2=0.9)
        final_model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
        return final_model
    
resnetmodel = get_resnet()
xceptionmodel = get_xception()
inceptionmodel = get_inception()
dense121model = get_dense121()
dense169model = get_dense169()
from keras.callbacks import *

es = EarlyStopping(monitor = 'val_loss',verbose = 1,mode='min')
plat = ReduceLROnPlateau(monitor = 'val_loss',verbose=1,factor=0.1,min_lr = 0.00001,patience=5)
Checkpoint= ModelCheckpoint("./resnetmodel_adam_0001.h5", monitor='val_loss', verbose=1, save_best_only=True,
       save_weights_only=True,mode='min')


train_history1 = resnetmodel.fit(
            train_dataset, 
            validation_data = valid_dataset, 
            steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,            
            validation_steps=valid_labels.shape[0] // BATCH_SIZE,            
            callbacks=[es,plat,Checkpoint],
            epochs=EPOCHS,
            verbose=1
            )
from keras.callbacks import *

es = EarlyStopping(monitor = 'val_loss',verbose = 1,mode='min')
plat = ReduceLROnPlateau(monitor = 'val_loss',verbose=1,factor=0.1,min_lr = 0.00001,patience=5)
Checkpoint= ModelCheckpoint("./xception_adam_0001.h5", monitor='val_loss', verbose=1, save_best_only=True,
       save_weights_only=True,mode='min')

train_history2 = xceptionmodel.fit(
            train_dataset, 
            validation_data = valid_dataset, 
            steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,            
            validation_steps=valid_labels.shape[0] // BATCH_SIZE,            
            callbacks=[es,plat,Checkpoint],
            epochs=EPOCHS,
            verbose=1
            )
from keras.callbacks import *

es = EarlyStopping(monitor = 'val_loss',verbose = 1,mode='min')
plat = ReduceLROnPlateau(monitor = 'val_loss',verbose=1,factor=0.1,min_lr = 0.00001,patience=5)
Checkpoint= ModelCheckpoint("./inception_adam_0001.h5", monitor='val_loss', verbose=1, save_best_only=True,
       save_weights_only=True,mode='min')

train_history3 = inceptionmodel.fit(
            train_dataset, 
            validation_data = valid_dataset, 
            steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,            
            validation_steps=valid_labels.shape[0] // BATCH_SIZE,            
            callbacks=[es,plat,Checkpoint],
            epochs=EPOCHS,
            verbose=1
            )
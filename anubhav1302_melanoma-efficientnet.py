!pip install -U efficientnet
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from efficientnet.tfkeras import *

import cv2

import tensorflow as tf

from tensorflow.keras.applications import *

from tensorflow.keras.models import *

from tensorflow.keras.layers import *

from tensorflow.keras.utils import Sequence,to_categorical

from sklearn.preprocessing import LabelEncoder,StandardScaler

from tensorflow.keras.callbacks import *

from tensorflow.keras.optimizers import *

from tensorflow.keras.losses import *

from tensorflow.keras.regularizers import l1

import matplotlib.pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.metrics import roc_auc_score

from tensorflow.keras.metrics import AUC

from sklearn.metrics import roc_auc_score

from sklearn.utils import shuffle

from tqdm import tqdm

from sklearn.model_selection import train_test_split,StratifiedKFold

from tensorflow.keras.utils import plot_model
train_images_path='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

test_images_path='/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

train_df=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

sample_sub=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
print('Train Data Shape: {}'.format(train_df.shape))

train_df.head()
print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

print('Running on TPU ', tpu.master())

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



strategy = tf.distribute.experimental.TPUStrategy(tpu)

print("REPLICAS: ", strategy.num_replicas_in_sync)



BATCH_SIZE = 16 * strategy.num_replicas_in_sync
gcs_path = KaggleDatasets().get_gcs_path()

def format_train_path(st):

    return gcs_path + '/jpeg/train/' + st + '.jpg'



def format_test_path(st):

    return gcs_path + '/jpeg/test/' + st + '.jpg'



train_data,val_data=train_test_split(train_df,test_size=0.1)

train_paths = train_data.image_name.apply(format_train_path).values

val_paths = val_data.image_name.apply(format_train_path).values



train_labels = train_data['target'].values

val_labels = val_data['target'].values

DIMS=(256,256,3)

EPOCHS=8
def decode_image(filename,label=None,image_size=(DIMS[0],DIMS[1])):

    bits=tf.io.read_file(filename)

    img=tf.image.decode_jpeg(bits,channels=3)

    img=tf.cast(img,tf.float32)/255.0

    img=tf.image.central_crop(img,0.3)

    img=tf.image.resize(img,image_size)

    if label is None:

        return img

    else:

        return img, label

    

def data_augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    image = tf.image.rot90(image)

    if label is None:

        return image

    else:

        return image, label

train_dataset=(tf.data.Dataset.from_tensor_slices((train_paths,train_labels)).map(decode_image,num_parallel_calls=AUTO)

               .map(data_augment,num_parallel_calls=AUTO).repeat()

              .shuffle(13)

              .batch(BATCH_SIZE).prefetch(AUTO))



val_dataset=(tf.data.Dataset.from_tensor_slices((val_paths,val_labels))

             .map(decode_image,num_parallel_calls=AUTO)

             .shuffle(13)

             .batch(BATCH_SIZE)

             .cache()

             .prefetch(AUTO))
with strategy.scope():

    inp=Input(DIMS)

    x=EfficientNetB7(include_top=False,input_tensor=inp)

    gap=SeparableConv2D(2048,2,activation='relu',padding='same')(x.output)

    

    x_0=Conv2D(512,(1,1),1,padding='same')(x.output)

    x_0=BatchNormalization()(x_0)

    x_0=Activation('relu')(x_0)

    

    x_1=Conv2D(512,(2,2),1,padding='same')(x.output)

    x_1=BatchNormalization()(x_1)

    x_1=Activation('relu')(x_1)

    

    x_2=Conv2D(512,(3,3),1,padding='same')(x.output)

    x_2=BatchNormalization()(x_2)

    x_2=Activation('relu')(x_2)

    

    x=Concatenate()([x_0,x_1,x_2,gap])

    x=Conv2D(2000,(2,2),strides=2,padding='same')(x)

    x=BatchNormalization()(x)

    x=Activation('relu')(x)

    x=GlobalAveragePooling2D()(x)

    

    out=Dense(1,activation='sigmoid')(x)    

    model=Model(inp,out) 

        

    model.compile(

        optimizer=Adam(0.001),

        loss = 'binary_crossentropy' ,

        metrics=[AUC()]

    )

STEPS_PER_EPOCH = len(train_labels) // BATCH_SIZE

mc=ModelCheckpoint('classifier.h5',monitor='val_loss',save_best_only=True,verbose=1,period=1)

rop=ReduceLROnPlateau(monitor='val_loss',min_lr=0.0000001,patience=2,mode='min')
history=model.fit(train_dataset,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,

                  validation_data=val_dataset,

                 callbacks=[mc])
def plot_metrics(metrics,name=['loss','AUC']):

    epochs = range(1, len(metrics[0]) + 1)

    plt.plot(epochs, metrics[0], 'b',color='red', label='Training '+name[0])

    plt.plot(epochs, metrics[1], 'b',color='blue', label='Validation '+name[0])

    plt.title('Metric Plot')

    plt.legend()

    plt.figure()

    plt.plot(epochs, metrics[2], 'b', color='red', label='Training '+name[1])

    plt.plot(epochs, metrics[3], 'b',color='blue', label='Validation '+name[1])

    plt.legend()

    plt.show()
plot_metrics([history.history['loss'],history.history['val_loss'],

              history.history['auc'],history.history['val_auc']])
test_paths = sample_sub.image_name.apply(format_test_path).values

test_dataset=(tf.data.Dataset.from_tensor_slices(test_paths)

             .map(decode_image,num_parallel_calls=AUTO)

             .batch(BATCH_SIZE))
model=load_model('classifier.h5')

preds=model.predict(test_dataset,verbose=1)

sample_sub['target'] = preds

sample_sub.to_csv('submission.csv', index=False)

sample_sub.head()
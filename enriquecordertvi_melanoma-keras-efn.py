!pip install -U efficientnet
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

import tensorflow as tf

from tensorflow.keras.applications import *

from tensorflow.keras.models import *

from tensorflow.keras.layers import *

from tensorflow.keras.utils import Sequence

from tensorflow.keras.callbacks import *

from tensorflow.keras.optimizers import *

import efficientnet.tfkeras as efn

import matplotlib.pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.metrics import roc_auc_score

from tensorflow.keras.metrics import AUC

from tqdm import tqdm

from sklearn.model_selection import train_test_split,StratifiedKFold
train_images_path='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

test_images_path='/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

train_df=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

sample_sub=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
print('Train Data Shape: {}'.format(train_df.shape))

train_df.head()
#Duplicate entries

print('Number of Unique ids: {}'.format(train_df['image_name'].nunique()))
train_df['target'].value_counts()
#target plotting

zero_targets=train_df['target'][train_df['target']==0].count()

ones_targets=train_df['target'][train_df['target']==1].count()

labels=['Class 0','Class 1']

t_circle=plt.Circle((0,0),0.7,color='white')

plt.pie([zero_targets,ones_targets], labels=labels, colors=['red','green'])

p=plt.gcf()

p.gca().add_artist(t_circle)

plt.show()
sns.catplot(x='sex',y='age_approx',data=train_df,hue='target')
def read_images(ix):

    img=cv2.imread(os.path.join(train_images_path,ix+'.jpg'))

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    return img
_,axs=plt.subplots(4,4,figsize=(13,13))

axs=axs.flatten()

for img_ix,lbl,ax in zip(train_df['image_name'],train_df['target'],axs):

    img=read_images(img_ix)

    ax.imshow(img)

    ax.set_title('Target: '.format(lbl))

plt.show()
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



train_data,val_data=train_test_split(train_df,test_size=0.2)



train_paths = train_data.image_name.apply(format_train_path).values

val_paths = val_data.image_name.apply(format_train_path).values



train_labels = train_data['target'].values

val_labels = val_data['target'].values
DIMS=(512,512,3)

EPOCHS=7
def decode_image(filename,label=None,image_size=(DIMS[0],DIMS[1])):

    bits=tf.io.read_file(filename)

    img=tf.image.decode_jpeg(bits,channels=3)

    img=tf.cast(img,tf.float32)/255.0

    img=tf.image.resize(img,image_size)

    if label is None:

        return img

    else:

        return img, label

    

def data_augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    image = tf.image.adjust_brightness(image,0.2)

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
def focal_loss(gamma=2., alpha=.25):

    def focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed
with strategy.scope():

    inp=Input(DIMS)

    base_feat_0=efn.EfficientNetB7(weights='imagenet',include_top=False,input_tensor=inp)

    base_feat_1=DenseNet169(weights='imagenet',include_top=False,input_tensor=inp)    

    

    x_0=GlobalAveragePooling2D()(base_feat_0.output)

    x_1=GlobalAveragePooling2D()(base_feat_1.output)

    x_1=Dense(2048)(x_1)

    x_1=LeakyReLU()(x_1)

    x_1=Dense(1024)(x_1)

    x_1=LeakyReLU()(x_1)

    

    x=Concatenate()([x_0,x_1])

    x=Dense(1024)(x)

    x=LeakyReLU()(x)

    

    x=Dense(512)(x)

    x=LeakyReLU()(x)

    

    out=Dense(1,activation='sigmoid')(x)

    model=Model(inp,out)

        

    model.compile(

        optimizer=Adam(),

        loss = 'binary_crossentropy',

        metrics=[AUC()]

    )
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE

mc=ModelCheckpoint('classifier.h5',monitor='val_loss',save_best_only=True,verbose=1,period=1)

rop=ReduceLROnPlateau(monitor='val_loss',min_lr=0.0000001,patience=2,mode='min')
history=model.fit(train_dataset,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,

                  validation_data=val_dataset,

                 callbacks=[mc,rop])
def plot_metrics(metrics,name=['loss','Acc']):

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
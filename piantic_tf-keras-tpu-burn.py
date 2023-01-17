!pip install -q efficientnet

!pip install tensorflow-addons==0.9.1    
import pandas as pd

import numpy as np

from glob import glob

import time

import random

import warnings



import keras

import tensorflow as tf



from tensorflow.keras.models import Sequential , Model

from tensorflow.keras.layers import Dense, Conv2D, ReLU, Dropout, Flatten, Activation, GlobalAveragePooling2D, AveragePooling2D

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler



from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications import DenseNet121

import tensorflow.keras.applications.inception_resnet_v2 as InceptionResnetV2

import tensorflow.keras.applications.inception_resnet_v2 as InceptionResnetV2



import os

import tensorflow as tf, tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import efficientnet.tfkeras as efn

from kaggle_datasets import KaggleDatasets



from tensorflow.keras.utils import plot_model



from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

from sklearn.model_selection import GroupKFold, StratifiedKFold



print(tf.__version__)

print(tf.keras.__version__)



warnings.simplefilter('ignore')
def seed_everything(seed=2020):

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    os.environ['TF_KERAS'] = '1'



seed = 2020

seed_everything(seed)
GCS_DS_PATH = KaggleDatasets().get_gcs_path('melanoma-merged-external-data-512x512-jpeg')

GCS_TRAIN_PATH = '/512x512-dataset-melanoma/512x512-dataset-melanoma/'

GCS_TEST_PATH = '/512x512-test/512x512-test/'



path='../input/siim-isic-melanoma-classification/'

#path='../input/melanoma-merged-external-data-512x512-jpeg/'



#train_df = pd.read_csv(path + 'train.csv')

test_df = pd.read_csv(path + 'test.csv')

sub = pd.read_csv(path + 'sample_submission.csv')
train_df = pd.read_csv('../input/melanoma-merged-external-data-512x512-jpeg/folds_08062020.csv')

train_df.head()
mc = ModelCheckpoint('best_model.h5',monitor='val_loss',mode='min',save_best_only=True,verbose=1)
es = EarlyStopping(monitor=tf.keras.metrics.AUC(),mode='max',verbose=1,patience=5)
def get_model_checkpoint(name='best_model'):

    model_checkpoint = ModelCheckpoint(name + '.h5', monitor=tf.keras.metrics.AUC(), 

                               mode='min', verbose=1,

                               save_best_only=True)

    

    return model_checkpoint
AUTO = tf.data.experimental.AUTOTUNE



try :

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU :',tpu.master())

except ValueError : 

    tpu = None

    

if tpu :

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else :

    strategy = tf.distribute.get_strategy()

    

print("Replicas :", strategy.num_replicas_in_sync)
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

EPOCHS = 2 # you can change epochs. e.g. 40

img_size = 512

nb_classes = 1
def decode_image(filename, label=None, image_size=(img_size,img_size)) :

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.bfloat16) / 255.0

    image = tf.image.resize(image, image_size)

    

    if label == None :

        return image

    else :

        return image, label
bool_random_brightness = False

bool_random_contrast = False

bool_random_hue = False

bool_random_saturation = False



def data_augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    if bool_random_brightness:

        image = tf.image.random_brightness(image,0.2,seed=seed)

    if bool_random_contrast:

        image = tf.image.random_contrast(image,0.6,1.4, seed=seed)

    if bool_random_hue:

        image = tf.image.random_hue(image,0.07,seed=seed)

    if bool_random_saturation:

        image = tf.image.random_saturation(image,0.5,1.5,seed=seed)

    

    if label is None:

        return image

    else:

        return image, label
fold_number = 0



train_paths = train_df[train_df.fold != fold_number].image_id.apply(lambda x : GCS_DS_PATH + GCS_TRAIN_PATH + x + '.jpg').values

train_labels = train_df[train_df.fold != fold_number].target

train_labels = tf.cast(train_labels, tf.float32)



valid_paths = train_df[train_df.fold == fold_number].image_id.apply(lambda x : GCS_DS_PATH + GCS_TRAIN_PATH + x + '.jpg').values

valid_labels = train_df[train_df.fold == fold_number].target

valid_labels = tf.cast(valid_labels, tf.float32)



test_paths = sub.image_name.apply(lambda x : GCS_DS_PATH + GCS_TEST_PATH + x + '.jpg').values
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .map(data_augment, num_parallel_calls=AUTO)

    .cache()

    .repeat()

    .shuffle(1024)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)
test_dataset=(

    tf.data.Dataset

    .from_tensor_slices(test_paths)

    .map(decode_image ,num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)
valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((valid_paths, valid_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)
def create_test_data(test_paths,aug=False):

    test_data = (

        tf.data.Dataset.from_tensor_slices(test_paths)

        .map(decode_image, num_parallel_calls = AUTO)

        .map(data_augment, num_parallel_calls=AUTO)

        .batch(BATCH_SIZE)

        .prefetch(AUTO)

    )

    

    if aug == True :

        test_data = test_data.map(data_augment ,num_parallel_calls = AUTO)

    return test_data
import matplotlib.pyplot as plt



LR_START = 0.0001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.0001

LR_RAMPUP_EPOCHS = 4

LR_SUSTAIN_EPOCHS = 6

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_scheduler = LearningRateScheduler(lrfn , verbose=True)



rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
import tensorflow_addons as tfa



bool_focal_loss = True

bool_label_smoothing = False



def get_model(name):



    if name == "EfficientNet":        

        base_model = efn.EfficientNetB0(weights='noisy-student',

                               include_top = False,

                               input_shape=(img_size,img_size,3)

                              )            

    elif name == 'ResNet':

        base_model = ResNet50(weights='imagenet',

                              include_top=False,

                              input_shape=(img_size,img_size,3))

    elif name == 'DenseNet':       

        base_model = DenseNet121(weights='imagenet',

                                 include_top=False,

                                 input_shape=(img_size,img_size,3))

    elif name == 'MobileNet':

        base_model = MobileNet(weights='imagenet', 

                               include_top=False,

                               input_shape=(img_size,img_size,3))

    elif name == 'IncepResnet':       

        base_model = InceptionResNetV2(weights='imagenet',

                                       include_top=False,

                                       input_shape=(img_size,img_size,3))

    elif name == 'CumstomModel':

        base_model = construct_model() # it need to be modified

          

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = L.Dense(1024,activation='relu')(x)

    x = L.Dropout(0.5)(x,training=True)

    predictions = Dense(1 ,activation='sigmoid')(x)

    

    if bool_focal_loss : 

        my_loss= tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)

    elif bool_label_smoothing :

        my_loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)

    else :

        my_loss = 'binary_crossentropy'

        

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss=my_loss, metrics=['accuracy', tf.keras.metrics.AUC()])  

    

    return model
with strategy.scope() : 

   # mc = get_model_checkpoint('EfficientNet')

    model_effnet = get_model("EfficientNet")



    history = model_effnet.fit(

        train_dataset, 

        epochs=EPOCHS, 

        callbacks=[lr_scheduler, mc],

        steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,

        validation_data=valid_dataset,

        validation_steps=valid_labels.shape[0] // BATCH_SIZE

    )
preds_effnet = model_effnet.predict(test_dataset, verbose=1)

sub.target = preds_effnet

sub.to_csv('sub_effnet.csv', index=False)

sub.head()
tta_num = 5

probabilities = []

for i in range(tta_num) :

    print('TTA number :',i+1)

    test_tta = create_test_data(test_paths)

    prob = model_effnet.predict(test_tta)

    probabilities.append(prob)

    

    

tab = np.zeros((len(probabilities[1]),1))

for i in range(len(probabilities[1])) :

    for j in range(tta_num) :

        tab[i] += probabilities[j][i] 

tab = tab / tta_num

sub['target'] = tab

sub.to_csv('efficientNet_TTA.csv',index=False)
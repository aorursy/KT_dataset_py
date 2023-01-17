!pip install knockknock #For sending message to telegram

!pip install -q efficientnet
import os

import cv2

import math

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,precision_score,recall_score,ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split,KFold,StratifiedKFold

from transformers import get_cosine_schedule_with_warmup

from albumentations import *

from kaggle_datasets import KaggleDatasets



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras import optimizers

import efficientnet.tfkeras as efn

from albumentations import *





from knockknock import telegram_sender 

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

token = user_secrets.get_secret("token")

chat_id = user_secrets.get_secret("chat_id")



import warnings  

warnings.filterwarnings('ignore')
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



GCS_DS_PATH = KaggleDatasets().get_gcs_path()
SEED = 42

BATCH_SIZE = 8 * strategy.num_replicas_in_sync

SIZE = [720,720]

LR = 0.0008

WEIGHT_DECAY = 0

EPOCHS = 35

WARMUP = 15

TTA = 4
def seed_everything(SEED):

    np.random.seed(SEED)

    tf.random.set_seed(SEED)



seed_everything(SEED)
DIR_INPUT = '/kaggle/input/plant-pathology-2020-fgvc7'

train_df = pd.read_csv(DIR_INPUT + '/train.csv')

test_df = pd.read_csv(DIR_INPUT + '/test.csv')

cols = list(train_df.columns[1:])
train,valid = train_test_split(train_df,test_size = 0.2,random_state = SEED)
transform = {

    'train' :Compose([

        Resize(SIZE[0],SIZE[1],always_apply=True),

        HorizontalFlip(p=0.5),

        VerticalFlip(p=0.5),

        RandomRotate90(p=0.5),

        Rotate(limit=25.0,p=0.8)]) }



def preprocess(df,test=False):

    paths = df.image_id.apply(lambda x: GCS_DS_PATH + '/images/' + x + '.jpg').values

    labels = df.loc[:,'healthy':].values

    if test==False:

        return paths,labels

    else:

        return paths

    

def decode_image(filename, label=None, image_size=(SIZE[0], SIZE[1])):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3) 

    image = tf.image.resize(image, image_size)

    image = tf.cast(image, tf.float32)

    image = tf.image.per_image_standardization(image)

    if label is None:

        return image

    else:

        return image, label

    

def data_augment(image, label=None, seed=SEED):

    image = tf.image.rot90(image,k=np.random.randint(4))

    image = tf.image.random_flip_left_right(image, seed=seed)

    image = tf.image.random_flip_up_down(image, seed=seed)

    if label is None:

        return image

    else:

        return image, label

    

def albu(image):

    transforms = transform['train']

    image = transforms(image=image.numpy())['image']

    image = tf.cast(image, tf.float32)

    return image

    

def albu_fn(image,label=None):

    [image,] = tf.py_function(albu, [image], [tf.float32])

    if label is None:

        return image

    else:

        return image, label
train_dataset = (tf.data.Dataset

    .from_tensor_slices(preprocess(train))

    .map(decode_image, num_parallel_calls=AUTO)

    .map(data_augment, num_parallel_calls=AUTO)

    .shuffle(SEED)

    .batch(BATCH_SIZE,drop_remainder=True)

    .repeat()

    .prefetch(AUTO))



valid_dataset = (tf.data.Dataset

    .from_tensor_slices(preprocess(valid))

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO))



test_dataset = (tf.data.Dataset

    .from_tensor_slices(preprocess(test_df,test=True))

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE))
def plot_transform(image_id,num_images=7):

    plt.figure(figsize=(30,10))

    path,_ = preprocess(train.iloc[image_id:image_id+1])

    for i in range(1,num_images+1):

        plt.subplot(1,num_images+1,i)

        plt.axis('off')

        image = decode_image(filename=path[0])

        image = data_augment(image=image)

        plt.imshow(image)



plot_transform(9)
with strategy.scope():

    model = tf.keras.Sequential([

        efn.EfficientNetB6(input_shape=(SIZE[0], SIZE[1], 3),weights='imagenet',pooling='avg',include_top=False),

        Dense(4, activation='softmax')

    ])

        

    model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy'])
def get_cosine_schedule_with_warmup(lr,num_warmup_steps, num_training_steps, num_cycles=0.5):

    """

    Modified the get_cosine_schedule_with_warmup from huggingface for tenserflow

    (https://huggingface.co/transformers/_modules/transformers/optimization.html#get_cosine_schedule_with_warmup)



    Create a schedule with a learning rate that decreases following the

    values of the cosine function between 0 and `pi * cycles` after a warmup

    period during which it increases linearly between 0 and 1.

    """



    def lrfn(epoch):

        if epoch < num_warmup_steps:

            return float(epoch) / float(max(1, num_warmup_steps)) * lr

        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr



    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



lr_schedule= get_cosine_schedule_with_warmup(lr=LR,num_warmup_steps=WARMUP,num_training_steps=EPOCHS)
STEPS_PER_EPOCH = train.shape[0] // BATCH_SIZE

@telegram_sender(token=token, chat_id=int(chat_id))

def train():

    history = model.fit(

        train_dataset, 

        epochs=EPOCHS, 

        callbacks=[lr_schedule],

        steps_per_epoch=STEPS_PER_EPOCH,

        validation_data=valid_dataset)

    

    string = 'Train acc:{:.4f} Train loss:{:.4f},Val acc:{:.4f} Val loss:{:.4f}'.format( \

        model.history.history['categorical_accuracy'][-1],model.history.history['loss'][-1],\

        model.history.history['val_categorical_accuracy'][-1],model.history.history['val_loss'][-1])

    

    return string
train()
def display_training_curves(training, validation, title, subplot):

    """

    Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu

    """

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(15,15), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])
display_training_curves(

    model.history.history['loss'], 

    model.history.history['val_loss'], 

    'loss', 211)

display_training_curves(

    model.history.history['categorical_accuracy'], 

    model.history.history['val_categorical_accuracy'], 

    'accuracy', 212)
test_pred = model.predict(test_dataset, verbose=1)

submission_df = pd.read_csv(DIR_INPUT + '/sample_submission.csv')

submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = test_pred

submission_df.to_csv('submission.csv', index=False)

pd.Series(np.argmax(submission_df[cols].values,axis=1)).value_counts()
test_pred_tta = np.zeros((len(test_df),4))

for i in range(TTA):

    test_dataset_tta = (tf.data.Dataset

    .from_tensor_slices(preprocess(test_df,test=True))

    .map(decode_image, num_parallel_calls=AUTO)

    .map(data_augment, num_parallel_calls=AUTO)    

    .batch(BATCH_SIZE))

    test_pred_tta += model.predict(test_dataset_tta, verbose=1)

submission_df = pd.read_csv(DIR_INPUT + '/sample_submission.csv')

submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = test_pred_tta/TTA

submission_df.to_csv('submission_tta.csv', index=False)

pd.Series(np.argmax(submission_df[cols].values,axis=1)).value_counts()
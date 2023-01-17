import pandas as pd

import numpy as np

import cv2

import os

from pathlib import Path

import matplotlib.pyplot as plt

import random

import json



import skimage.io

from sklearn.preprocessing import OneHotEncoder #One-hot 인코더

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score



import tensorflow as tf

from tensorflow.keras import Model, Sequential

from tensorflow.keras.models import load_model

from tensorflow.keras.utils import Sequence

from tensorflow.keras.layers import *

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.preprocessing.image import ImageDataGenerator



from tensorflow.keras.applications import ResNet50



import albumentations as albu
DATA_PATH = '../input/prostate-cancer-grade-assessment'

MODELS_PATH = '.'

IMG_SIZE = 56 

SEQ_LEN = 16 # 16^2 = 256

BATCH_SIZE = 16

MDL_VERSION = 'v18'

SEED = 80
class DataGenPanda(Sequence):

    #initialize

    def __init__(self, imgs_path, df, batch_size=32, 

                 mode='fit', shuffle=False, aug=None, 

                 seq_len=12, img_size=128, n_classes=6):

        self.imgs_path = imgs_path

        self.df = df

        self.shuffle = shuffle

        self.mode = mode

        self.aug = aug

        self.batch_size = batch_size

        self.img_size = img_size

        self.seq_len = seq_len

        self.n_classes = n_classes

        self.side = int(seq_len ** .5)

        self.on_epoch_end()

    

    def __len__(self):

        return int(np.floor(len(self.df) / self.batch_size))

    

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.df))

        if self.shuffle:

            np.random.shuffle(self.indexes)

            

    def __getitem__(self, index):

        X = np.zeros((self.batch_size, self.side * self.img_size, self.side * self.img_size, 3), dtype=np.float32)

        imgs_batch = self.df[index * self.batch_size : (index + 1) * self.batch_size]['image_id'].values

        for i, img_name in enumerate(imgs_batch):

            img_path = '{}/{}.tiff'.format(self.imgs_path, img_name)

            img_patches = self.get_patches(img_path)

            X[i, ] = self.glue_to_one(img_patches)

        if self.mode == 'fit':

            y = np.zeros((self.batch_size, self.n_classes), dtype=np.float32)

            lbls_batch = self.df[index * self.batch_size : (index + 1) * self.batch_size]['isup_grade'].values

            for i in range(self.batch_size):

                y[i, lbls_batch[i]] = 1

            return X, y

        elif self.mode == 'predict':

            return X

        else:

            raise AttributeError('mode parameter error')

            

    def get_patches(self, img_path):

        num_patches = self.seq_len

        p_size = self.img_size

        img = skimage.io.MultiImage(img_path)[-1] / 255

        if self.aug:

            img = self.aug(image=img)['image'] 

        pad0, pad1 = (p_size - img.shape[0] % p_size) % p_size, (p_size - img.shape[1] % p_size) % p_size

        img = np.pad(

            img,

            [

                [pad0 // 2, pad0 - pad0 // 2], 

                [pad1 // 2, pad1 - pad1 // 2], 

                [0, 0]

            ],

            constant_values=1

        )

        img = img.reshape(img.shape[0] // p_size, p_size, img.shape[1] // p_size, p_size, 3)

        img = img.transpose(0, 2, 1, 3, 4).reshape(-1, p_size, p_size, 3)

        if len(img) < num_patches:

            img = np.pad(

                img, 

                [

                    [0, num_patches - len(img)],

                    [0, 0],

                    [0, 0],

                    [0, 0]

                ],

                constant_values=1

            )

        idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:num_patches]

        return np.array(img[idxs])

    def glue_to_one(self, imgs_seq):

        img_glue = np.zeros((self.img_size * self.side, self.img_size * self.side, 3), dtype=np.float32)

        #img_size * side 만큼 float32비트 형식으로 0으로 채움

        

        for i, ptch in enumerate(imgs_seq):

            x = i // self.side

            y = i % self.side

            img_glue[x * self.img_size : (x + 1) * self.img_size, 

                     y * self.img_size : (y + 1) * self.img_size, :] = ptch

        return img_glue
#data augmentation 수행

aug = albu.Compose(

    [

        albu.HorizontalFlip(p=.25),

        albu.VerticalFlip(p=.25),

        albu.ShiftScaleRotate(shift_limit=.1, scale_limit=.1, rotate_limit=20, p=.25)

    ]

)
def kappa_score(y_true, y_pred):

    

    y_true=tf.math.argmax(y_true)

    y_pred=tf.math.argmax(y_pred)

    return tf.compat.v1.py_func(cohen_kappa_score ,(y_true, y_pred),tf.double)
pre_file = '../input/panda-resnet50-model-1/model_v18_resnet50.h5'

resnet_weights_path = '../input/panda-resnet50-model-1/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

input = Input(shape=(224, 224, 3))

model = Sequential()



bottleneck = ResNet50(weights=resnet_weights_path, include_top=False, pooling='avg', input_tensor=input)

model.add(bottleneck)

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(.25))

model.add(Dense(6, activation='sigmoid'))

model.summary()



model.compile(

    loss='binary_crossentropy',

    optimizer=Adam(lr=1e-5),

    metrics=['binary_crossentropy',kappa_score]

)

model.load_weights(pre_file)
test = pd.read_csv('{}/test.csv'.format(DATA_PATH))

preds = [[0] * 6] * len(test)

if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

    subm_datagen = DataGenPanda(

        imgs_path='{}/test_images'.format(DATA_PATH), 

        df=test,

        batch_size=1,

        mode='predict', 

        shuffle=False, 

        aug=None, 

        seq_len=SEQ_LEN, 

        img_size=IMG_SIZE, 

        n_classes=6

    )

    preds = model.predict_generator(subm_datagen)

    print('preds done, total:', len(preds))

else:

    print('preds are zeros')

test['isup_grade'] = np.argmax(preds, axis=1)

test.drop('data_provider', axis=1, inplace=True)

test.to_csv('submission.csv', index=False)

print('submission saved')
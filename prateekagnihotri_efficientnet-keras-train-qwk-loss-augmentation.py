# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import glob
glob.glob('..s/input/prostate-cancer-grade-assessment/*')
len(glob.glob('../input/prostate-cancer-grade-assessment/train_images/*.tiff'))
!pip install -qq ../input/efficientnet/efficientnet-1.0.0-py3-none-any.whl
import os
import cv2
import time
import skimage.io
import numpy as np
import pandas as pd
import imgaug as ia
from PIL import Image
from tqdm import tqdm
from random import shuffle
import matplotlib.pyplot as plt
import efficientnet.keras as efn
from imgaug import augmenters as iaa

from sklearn.model_selection import train_test_split

from keras import Model
import keras.backend as K
from keras.layers import *
from keras.optimizers import Adam, RMSprop
from keras.applications.nasnet import  preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
def quadratic_kappa_coefficient(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    n_classes = K.cast(y_pred.shape[-1], "float32")
    weights = K.arange(0, n_classes, dtype="float32") / (n_classes - 1)
    weights = (weights - K.expand_dims(weights, -1)) ** 2

    hist_true = K.sum(y_true, axis=0)
    hist_pred = K.sum(y_pred, axis=0)

    E = K.expand_dims(hist_true, axis=-1) * hist_pred
    E = E / K.sum(E, keepdims=False)

    O = K.transpose(K.transpose(y_true) @ y_pred)  # confusion matrix
    O = O / K.sum(O)

    num = weights * O
    den = weights * E

    QWK = (1 - K.sum(num) / K.sum(den))
    return QWK

def quadratic_kappa_loss(scale=2.0):
    def _quadratic_kappa_loss(y_true, y_pred):
        QWK = quadratic_kappa_coefficient(y_true, y_pred)
        loss = -K.log(K.sigmoid(scale * QWK))
        return loss
        
    return _quadratic_kappa_loss
I = Input((256,256,3))
efnb3 = efn.EfficientNetB3(weights = None, include_top = False, input_tensor = I, pooling = 'avg', classes = None)
for layer in efnb3.layers:
    layer.trainable = True
x = Dropout(0.5)(efnb3.output)
x = Dense(64, activation='relu')(x)
x = Dense(6,activation='softmax')(x)

model = Model(inputs = efnb3.input, outputs = x)

model.compile(optimizer=Adam(1e-4), loss=quadratic_kappa_loss(scale=6.0), metrics=['acc',quadratic_kappa_coefficient])
# model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['acc'])
# model.summary()
def get_seq():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10), # rotate by -45 to +45 degrees
                shear=(-5, 5), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
                    ]),
                    iaa.Invert(0.01, per_channel=True), # invert color channels
                    iaa.Add((-2, 2), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.9, 1.1), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-1, 0),
                            first=iaa.Multiply((0.9, 1.1), per_channel=True),
                            second=iaa.ContrastNormalization((0.9, 1.1))
                        )
                    ]),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_image(img_name):
    data_dir = '../input/prostate-cancer-grade-assessment/train_images'
    img_path = os.path.join(data_dir, f'{img_name}.tiff')
    img = skimage.io.MultiImage(img_path)
    img = cv2.resize(img[-1], (256,256))
    return img

def data_gen(list_files, id_label_map, batch_size, augment=False):
    seq = get_seq()
    while True:
        shuffle(list_files)
        for batch in chunker(list_files, batch_size):
            X = [get_image(x) for x in batch]
            Y = np.zeros((len(batch),6))
            for i in range(len(batch)):
                Y[i,id_label_map[get_id_from_file_path(batch[i])]] = 1.0
            if augment:
                X = seq.augment_images(X)
            X = [preprocess_input(x) for x in X]

            yield np.array(X), np.array(Y)
def get_id_from_file_path(file_path):
    return file_path.split(os.path.sep)[-1].replace('.tiff', '')
batch_size=32
df_train = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")
id_label_map = {k:v for k,v in zip(df_train.image_id.values, df_train.isup_grade.values)}
labeled_files = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv").image_id.values
test_files = pd.read_csv("../input/prostate-cancer-grade-assessment/test.csv").image_id.values

train, val = train_test_split(labeled_files, test_size=0.1, random_state=101010)
check_point = ModelCheckpoint('./model.h5',monitor='val_loss',verbose=True, save_best_only=True, save_weights_only=True)
early_stop = EarlyStopping(monitor='val_loss',patience=5,verbose=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1)
history = model.fit_generator(
    data_gen(train, id_label_map, batch_size, augment=True),
    validation_data=data_gen(val, id_label_map, batch_size),
    epochs=15, verbose=1,
    callbacks=[check_point,early_stop,reduce_lr],
    steps_per_epoch=len(train) // batch_size,
    validation_steps=len(val) // batch_size)
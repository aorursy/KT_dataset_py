# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load 



import warnings

warnings.filterwarnings(action='ignore')



import cv2 # image processing

import time, gc

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

from tqdm.auto import tqdm

import albumentations as A



# deep learning frameworks

import tensorflow as tf

from tensorflow import keras

from keras.models import Model

from keras.optimizers import Adam

from keras.models import clone_model

from keras.utils.vis_utils import plot_model

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input,Reshape,Concatenate, concatenate,GaussianDropout 





# data visualization/serialization packages

import pickle

import seaborn as sns

import matplotlib.image as mpimg

from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

test_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
IMG_SIZE=64

N_CHANNELS=1

HEIGHT = 137

WIDTH = 236

SIZE = 64

CROP_SIZE = 64
def resize(df, size=64, need_progress_bar=True):

    resized = {}

    resize_size=64

    angle=0

    if need_progress_bar:

        for i in tqdm(range(df.shape[0])):

            #Reshape

            image=df.loc[df.index[i]].values.reshape(137,236)

            

            #Centering

            image_center = tuple(np.array(image.shape[1::-1]) / 2)

            matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)

            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,

                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            

            #Scaling

            matrix = cv2.getRotationMatrix2D(image_center, 0, 1.0)

            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,

                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            

            #Threshold and Contours

            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]



            idx = 0 

            ls_xmin = []

            ls_ymin = []

            ls_xmax = []

            ls_ymax = []

            for cnt in contours:

                idx += 1

                x,y,w,h = cv2.boundingRect(cnt)

                ls_xmin.append(x)

                ls_ymin.append(y)

                ls_xmax.append(x + w)

                ls_ymax.append(y + h)

            xmin = min(ls_xmin)

            ymin = min(ls_ymin)

            xmax = max(ls_xmax)

            ymax = max(ls_ymax)



            roi = image[ymin:ymax,xmin:xmax]

            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)



            resized[df.index[i]] = resized_roi.reshape(-1)

        

    else:

        for i in range(df.shape[0]):

            #Reshape

            image=df.loc[df.index[i]].values.reshape(137,236)

            

            #Centering

            image_center = tuple(np.array(image.shape[1::-1]) / 2)

            matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)

            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,

                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            

            #Scaling

            matrix = cv2.getRotationMatrix2D(image_center, 0, 1.0)

            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,

                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            

            #Threshold and Contours

            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]



            idx = 0 

            ls_xmin = []

            ls_ymin = []

            ls_xmax = []

            ls_ymax = []

            for cnt in contours:

                idx += 1

                x,y,w,h = cv2.boundingRect(cnt)

                ls_xmin.append(x)

                ls_ymin.append(y)

                ls_xmax.append(x + w)

                ls_ymax.append(y + h)

            xmin = min(ls_xmin)

            ymin = min(ls_ymin)

            xmax = max(ls_xmax)

            ymax = max(ls_ymax)



            roi = image[ymin:ymax,xmin:xmax]

            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)



            resized[df.index[i]] = resized_roi.reshape(-1)

            

    resized = pd.DataFrame(resized).T

    return resized
from keras.applications import Xception



# load the model

inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 1))

model = Conv2D(filters=3, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1))(inputs)

model = BatchNormalization(momentum=0.15)(model)

model = GaussianDropout(rate=0.3)(model)

fmodel = Conv2D(filters=3, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

fmodel = BatchNormalization(momentum=0.15)(fmodel)



# Pretrained model

base_model = Xception(weights=None, include_top=False)(fmodel)



model = BatchNormalization(momentum=0.15)(base_model)

model = GaussianDropout(rate=0.3)(model)

model = Flatten()(base_model)



model = Dense(1024, activation = "relu")(model)

model = BatchNormalization(momentum=0.15)(model)

model = GaussianDropout(rate=0.3)(model)

dense = Dense(512, activation = "relu")(model)

dense = BatchNormalization(momentum=0.15)(dense)

head_root = Dense(168, activation = 'softmax')(dense)



head_root = Dense(168, activation = 'softmax')(dense)

head_vowel = Dense(11, activation = 'softmax')(dense)

head_consonant = Dense(7, activation = 'softmax')(dense)



model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])

    

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
class MultiOutputDataGenerator(keras.preprocessing.image.ImageDataGenerator):



    def flow(self,

             x,

             y=None,

             batch_size=32,

             shuffle=True,

             sample_weight=None,

             seed=None,

             save_to_dir=None,

             save_prefix='',

             save_format='png',

             subset=None):



        targets = None

        target_lengths = {}

        ordered_outputs = []

        for output, target in y.items():

            if targets is None:

                targets = target

            else:

                targets = np.concatenate((targets, target), axis=1)

            target_lengths[output] = target.shape[1]

            ordered_outputs.append(output)





        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,

                                         shuffle=shuffle):

            target_dict = {}

            i = 0

            for output in ordered_outputs:

                target_length = target_lengths[output]

                target_dict[output] = flowy[:, i: i + target_length]

                i += target_length



            yield flowx, target_dict
model.load_weights("/kaggle/input/bengaliainoaugxceptionnet/model_noaug_v1.h5")



preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}



components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder

for i in range(4):

    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    df_test_img.set_index('image_id', inplace=True)



    X_test = resize(df_test_img, need_progress_bar=False)/255

    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    

    preds = model.predict(X_test)



    for i, p in enumerate(preds_dict):

        preds_dict[p] = np.argmax(preds[i], axis=1)



    for k,id in enumerate(df_test_img.index.values):  

        for i,comp in enumerate(components):

            id_sample=id+'_'+comp

            row_id.append(id_sample)

            target.append(preds_dict[comp][k])

    del df_test_img

    del X_test

    gc.collect()



df_sample = pd.DataFrame(

    {

        'row_id': row_id,

        'target':target

    },

    columns = ['row_id','target'] 

)

df_sample.to_csv('submission.csv',index=False)

df_sample.head()
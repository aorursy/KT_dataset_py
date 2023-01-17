# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt



import os, time, random, glob, pickle, warnings, math



from keras import backend as K

import tensorflow as tf

from keras.backend.tensorflow_backend import set_session

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.applications.resnet50 import  ResNet50

from keras.applications.densenet import  DenseNet121

from keras.layers import GlobalMaxPooling2D,Conv2D, Dense,BatchNormalization,Flatten,Input, Multiply ,Dropout,GlobalAveragePooling2D

from keras.utils import Sequence

from keras.optimizers import Adam

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

import cv2

warnings.filterwarnings('ignore') # warningを非表示にする

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # kerasのwarningを非表示にする

import keras.models

!pip install ../input/efficientnet-keras-source-code/repository/qubvel-efficientnet-c993591

import efficientnet.keras as efn 



model = keras.models.load_model('/kaggle/input/simple-se-resnet50/model_se_resnet50-30.h5', compile=False)
SEED = 2020

batch_size = 12 

dim = (125, 125)

SIZE = 125

stats = (0.0692, 0.2051)

HEIGHT = 137 

WIDTH = 236

from tqdm import tqdm

def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(img0, size=125, pad=16):

    #crop a box around pixels large than the threshold 

    #some images contain line at the sides

    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)

    #cropping may cut too much, so we need to add it back

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH

    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT

    img = img0[ymin:ymax,xmin:xmax]

    #remove lo intensity pixels as noise

    img[img < 28] = 0

    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly) + pad

    #make sure that the aspect ratio is kept in rescaling

    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    

    return cv2.resize(img,(size,size))
import gc

def test_batch_generator(df, batch_size):

    num_imgs = len(df)



    for batch_start in range(0, num_imgs, batch_size):

        curr_batch_size = min(num_imgs, batch_start + batch_size) - batch_start

        idx = np.arange(batch_start, batch_start + curr_batch_size)



        names_batch = df.iloc[idx, 0].values

        imgs_batch = 255 - df.iloc[idx, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

        X_batch = np.zeros((curr_batch_size, SIZE, SIZE, 1))

        

        for j in range(curr_batch_size):

            img = (imgs_batch[j,]*(255.0/imgs_batch[j,].max())).astype(np.uint8)

            img = crop_resize(img, size=SIZE)

            img = img[:, :, np.newaxis]

            X_batch[j,] = img



        yield X_batch, names_batch

TEST = [

    "../input/bengaliai-cv19/test_image_data_0.parquet",

    "../input/bengaliai-cv19/test_image_data_1.parquet",

    "../input/bengaliai-cv19/test_image_data_2.parquet",

    "../input/bengaliai-cv19/test_image_data_3.parquet",

]



# placeholders 

row_id = []

target = []



# iterative over the test sets

for fname in tqdm(TEST):

    test_ = pd.read_parquet(fname)

    test_gen = test_batch_generator(test_, batch_size=batch_size)



    for batch_x, batch_name in test_gen:

        batch_predict = model.predict(batch_x)

        for idx, name in enumerate(batch_name):

            row_id += [

                f"{name}_consonant_diacritic",

                f"{name}_grapheme_root",

                f"{name}_vowel_diacritic",

            ]

            target += [

                np.argmax(batch_predict[2], axis=1)[idx],

                np.argmax(batch_predict[0], axis=1)[idx],

                np.argmax(batch_predict[1], axis=1)[idx],

            ]



    del test_

    gc.collect()

    

    

df_sample = pd.DataFrame(

    {

        'row_id': row_id,

        'target':target

    },

    columns = ['row_id','target'] 

)



df_sample.to_csv('submission.csv',index=False)

gc.collect()
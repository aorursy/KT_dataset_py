# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input

from keras.optimizers import Adam

from keras.models import Model



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

test_df = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
train_df.head()
test_df.head()
sample_sub_df.head()
print(f'Size of training data: {train_df.shape}')

print(f'Size of test data: {test_df.shape}')
import math

import numpy as np

import h5py

import matplotlib.pyplot as plt

import scipy

from PIL import Image

from scipy import ndimage

import tensorflow as tf

from tensorflow.python.framework import ops

from tqdm.auto import tqdm

from glob import glob

import time, gc

import cv2





%matplotlib inline

np.random.seed(1)
y_grapheme_root=train_df["grapheme_root"]

y_vowel_diacritic=train_df["vowel_diacritic"]

y_cons_diacritic=train_df["consonant_diacritic"]
print(y_grapheme_root.max())

print(y_vowel_diacritic.max())

print(y_cons_diacritic.max())
def convert_to_one_hot(Y, C):

    Y = np.eye(C)[np.reshape(Y,-1)]

    return Y
Y_root = convert_to_one_hot(y_grapheme_root, y_grapheme_root.max()+1).T

Y_cons = convert_to_one_hot(y_cons_diacritic,y_cons_diacritic.max()+1).T

Y_vowel = convert_to_one_hot(y_vowel_diacritic, y_vowel_diacritic.max()+1).T
IMG_SIZE=64

N_CHANNELS=1
inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 1))

model = Conv2D(filters=8, kernel_size=(4,4), padding='SAME',strides = [1,1] ,activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1))(inputs)

model = MaxPool2D(pool_size=(4,4),strides=[4,4],padding='SAME')(model)

model = Flatten()(model)

head_root = Dense(168, activation = None)(model)

head_vowel = Dense(11, activation = None)(model)

head_consonant = Dense(7, activation = None)(model)



model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])



model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
batch_size = 64

epochs = 100
Y_root=Y_root.T

Y_cons =Y_cons.T

Y_vowel=Y_vowel.T
def resize(df, size=64, need_progress_bar=True):

    resized = {}

    resize_size=64

    if need_progress_bar:

        for i in tqdm(range(df.shape[0])):

            image=df.loc[df.index[i]].values.reshape(137,236)

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

            #image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

            image=df.loc[df.index[i]].values.reshape(137,236)

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
train_df_=pd.DataFrame()

for i in range(4):

    train_df_= pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train_df, on='image_id')

    print(train_df_.shape)

    X_train = train_df_.drop(['image_id','grapheme_root', 'vowel_diacritic', 'consonant_diacritic','grapheme'], axis=1)

    X_train=resize(X_train)/255

    X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    print(X_train.shape)

    model.fit(X_train,{'dense_1': Y_root[i*50210:(i+1)*50210,:], 'dense_2': Y_vowel[i*50210:(i+1)*50210,:], 'dense_3': Y_cons[i*50210:(i+1)*50210,:]},batch_size=batch_size,epochs = epochs)

    print(i)
del train_df,Y_root,Y_cons ,Y_vowel

print(train_df_.shape)

del train_df_
test_df.head()
preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}
components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder
test_df_=pd.DataFrame()

for i in range(4):

    test_df_= pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    test_df_.set_index('image_id', inplace=True)

    X_test=resize(test_df_)/255

    print(X_test.shape)

    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    print(X_test.shape)

    preds=model.predict(X_test)

    print(preds)

    for i, p in enumerate(preds_dict):

        preds_dict[p] = np.argmax(preds[i], axis=1)

        

    for k,id in enumerate(test_df_.index.values):  

        for i,comp in enumerate(components):

            id_sample=id+'_'+comp

            row_id.append(id_sample)

            target.append(preds_dict[comp][k])
df_sample = pd.DataFrame(

    {

        'row_id': row_id,

        'target':target

    },

    columns = ['row_id','target'] 

)

df_sample.to_csv('submission.csv',index=False)

df_sample.head()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import tensorflow as tf

import tensorflow.keras as keras

import cv2

import sys

import glob

import gc

import matplotlib.pyplot as plt

%matplotlib inline
input_path = '/kaggle/input/deepfake-detection-challenge/'

train_dir = glob.glob(input_path+'train_sample_videos/*.mp4')
meta = pd.read_json(input_path+'train_sample_videos/metadata.json').T

meta.head()
gc.collect()

len(train_dir)
x = meta['label'].value_counts().index

y = meta['label'].value_counts().values

plt.bar(x,y)

plt.show()
REAL = meta[meta['label'] == 'REAL'].index.values

FAKE_LIST = np.random.choice(meta[meta['label'] == 'FAKE'].index.values,len(REAL))

file_list = list(REAL) + list(FAKE_LIST) 
IMG_SHAPE=(229,229,3)
googlenet_base = keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
# cap = cv2.VideoCapture(train_dir[0])

# count = 0

# while count<1:

#     ret, frame = cap.read()

#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     img = keras.preprocessing.image.img_to_array(img)

#     img = cv2.resize(img, (229,229))

#     img = np.expand_dims(img, axis=0)

#     val = googlenet_base(img)

#     print(val.shape)

#     count += 1
def creat_features(train_dir, test=False):

    

    img_data = {}

    for files in train_dir:

        key = files.split('/')[5]

        

        if test:

            data_f = file_list + [ x.split('/')[5] for x in train_dir]

        else:

            data_f = file_list

        if key in data_f:

            cap = cv2.VideoCapture(files)

            count = 0

            if test is False:

                label = meta.loc[key]['label']

            print(f'processing video {key}')

            data = []

            while count<20:

                ret, frame = cap.read()

                if ret == False:

                    break;

                img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

                img = keras.preprocessing.image.img_to_array(img)

                img = cv2.resize(img, (229,229))

                #print(img.shape)

                img = np.expand_dims(img, axis=0)

                img = googlenet_base(img)

                data.append(img.numpy())

                gc.collect()

                count += 1

                #print(img.shape)

            if key in img_data:

                if test is False:

                    img_data[key].append([data, label])

                else:

                    img_data[key].append(data)

            else:

                img_data[key]=[]

                if test is False:

                    img_data[key].append([data, label])

                else:

                    img_data[key].append(data)

            cap.release()

    return img_data
train_features = creat_features(train_dir)

gc.collect()
def create_df(df_m, test=False):

    matrix = []

    label =[]

    for key in df_m:

        if test:

            temp = np.asarray(df_m[key][0])

            matrix.extend(temp.reshape((temp.shape[0],-1) ))

        else:

            temp = np.asarray(df_m[key][0][0])

            matrix.extend(temp.reshape((temp.shape[0],-1) ))

            label.extend([df_m[key][0][1]]*20)

    return matrix, label
X,y = create_df(train_features)
test_dir = glob.glob(input_path+'test_videos/*.mp4')

test_features = creat_features(test_dir, test=True)

X_test,_ = create_df(test_features, test=True)
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
m = SVC()

l = LabelEncoder()

t = l.fit_transform(y)

random = RandomForestClassifier()
random.fit(X, t)
predict = random.predict(X_test)
predict.shape
temp_df = pd.DataFrame()
temp_df['filename'] = test_features.keys()

temp_df['label'] = np.array(400*[0]).astype(int)
temp_df.dtypes
from collections import Counter

start = 0

delta = 20

for feat in test_features.keys():

    end = start+delta

    results = predict[start:end]

    temp_df.loc[temp_df['filename']==feat,'label'] = Counter(results).most_common(1)[0][0].astype(int)

    start = end
temp_df.head()
temp_df.to_csv('submission.csv', index=False)
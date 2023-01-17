import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import sys;

import hashlib;

from os.path import isfile

from joblib import Parallel, delayed

import psutil
train_df = pd.read_csv("../input/train.csv")

print(train_df.shape)

test_df = pd.read_csv("../input/sample_submission.csv")

test_df['diagnosis'] = np.nan

train = train_df.append(test_df)
def expand_path(p):

    if isfile('../input/train_images/' + p + '.png'): return '../input/train_images/' + p + '.png'

    if isfile('../input/test_images/' + p + '.png'): return '../input/test_images/' + p + '.png'

    return p

def getImageMetaData(p):

    strFile = expand_path(p)

    file = None;

    bRet = False;

    strMd5 = "";

    

    try:

        file = open(strFile, "rb");

        md5 = hashlib.md5();

        strRead = "";

        

        while True:

            strRead = file.read(8096);

            if not strRead:

                break;

            md5.update(strRead);

        #read file finish

        bRet = True;

        strMd5 = md5.hexdigest();

    except:

        bRet = False;

    finally:

        if file:

            file.close()



    return p,strMd5
img_meta_l = Parallel(n_jobs=psutil.cpu_count(), verbose=1)(

    (delayed(getImageMetaData)(fp) for fp in train.id_code))
img_meta_df = pd.DataFrame(np.array(img_meta_l))

img_meta_df.columns = ['id_code', 'strMd5']
train = train.merge(img_meta_df,on='id_code')
train['strMd5_count'] = train.groupby('strMd5').id_code.transform('count')
train['strMd5_train_count'] = train['strMd5'].map(train.groupby('strMd5')['diagnosis'].apply(lambda x:x.notnull().sum()))
train['strMd5_nunique'] = train.groupby('strMd5')['diagnosis'].transform('nunique').astype('int')
train.to_csv('strMd5.csv',index=None)
train[train.strMd5_count>1].strMd5_count.value_counts()
import matplotlib.pyplot as plt

import cv2
train[(train.strMd5_train_count>1)&(train.strMd5_nunique==1)].strMd5_count.value_counts()
strMd51 = train[(train.strMd5_count>1)&(train.strMd5_nunique==1)].strMd5.unique()

strMd5 = strMd51[0]

size = len(train[train['strMd5'] == strMd5]['id_code'])

fig = plt.figure(figsize = (20, 5))

for idx, img_name in enumerate(train[train['strMd5'] == strMd5]['id_code'][:size]):

    y = fig.add_subplot(1, size, idx+1)

    img = cv2.imread(expand_path(img_name))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    class_id = train[train.id_code==img_name]['diagnosis'].values

    y.set_title(img_name+f'Label: {class_id}')

    y.imshow(img)

plt.show()
train[(train.strMd5_count>1)&(train.strMd5_nunique>1)].strMd5_count.value_counts()
strMd52 = train[(train.strMd5_count>1)&(train.strMd5_nunique>1)].strMd5.unique()

strMd5 = strMd52[0]

for strMd5 in strMd52[:5]:

    size = len(train[train['strMd5'] == strMd5]['id_code'])

    fig = plt.figure(figsize = (20, 5))

    for idx, img_name in enumerate(train[train['strMd5'] == strMd5]['id_code'][:size]):

        y = fig.add_subplot(1, size, idx+1)

        img = cv2.imread(expand_path(img_name))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        class_id = train[train.id_code==img_name]['diagnosis'].values

        y.set_title(img_name+f'Label: {class_id}')

        y.imshow(img)

    plt.show()
train[(train.strMd5_count>1)&(train.diagnosis.isnull())].shape[0]
strMd52 = train[(train.strMd5_count>1)&(train.diagnosis.isnull())].strMd5.unique()

strMd5 = strMd52[0]

for strMd5 in strMd52[:5]:

    size = len(train[train['strMd5'] == strMd5]['id_code'])

    fig = plt.figure(figsize = (20, 5))

    for idx, img_name in enumerate(train[train['strMd5'] == strMd5]['id_code'][:size]):

        y = fig.add_subplot(1, size, idx+1)

        img = cv2.imread(expand_path(img_name))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        class_id = train[train.id_code==img_name]['diagnosis'].values

        y.set_title(img_name+f'Label: {class_id}')

        y.imshow(img)

    plt.show()
train[(train.strMd5_count==2)]['strMd5_train_count'].value_counts()
strMd52 = train[(train.strMd5_count>2)].strMd5.unique()

strMd5 = strMd52[0]

for strMd5 in strMd52:

    size = len(train[train['strMd5'] == strMd5]['id_code'])

    fig = plt.figure(figsize = (20, 5))

    for idx, img_name in enumerate(train[train['strMd5'] == strMd5]['id_code'][:size]):

        y = fig.add_subplot(1, size, idx+1)

        img = cv2.imread(expand_path(img_name))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        class_id = train[train.id_code==img_name]['diagnosis'].values

        y.set_title(img_name+f'Label: {class_id}')

        y.imshow(img)

    plt.show()
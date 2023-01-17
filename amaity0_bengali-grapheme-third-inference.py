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
import cv2

import zipfile

from tqdm import tqdm_notebook as tqdm

import random

import torch

import torchvision



SEED = 42

LABELS = 'train.csv'



import fastai

from fastai.vision import *

import warnings

warnings.filterwarnings("ignore")



fastai.__version__
HEIGHT = 137

WIDTH = 236

SIZE = (128,128)

BATCH = 64



PATH = '/kaggle/input/bengaliai-cv19/'



TEST = [PATH+'test_image_data_0.parquet',

        PATH+'test_image_data_1.parquet',

        PATH+'test_image_data_2.parquet',

        PATH+'test_image_data_3.parquet']



TRAIN = [PATH+'train_image_data_0.parquet',

         PATH+'train_image_data_1.parquet',

         PATH+'train_image_data_2.parquet',

         PATH+'train_image_data_3.parquet']



df_test = pd.read_csv(PATH+'test.csv')

df_test.describe()
#!mkdir '/kaggle/working/test'
def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(img0, size=SIZE, pad=16):

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

    return cv2.resize(img,size)
learn = load_learner(Path('/kaggle/input/bgraph-rn50-model'),'try2A-rn50-im128-wcv.pkl')

#preds, _ = learn.get_preds(ds_type=DatasetType.Test)
# def fillLabels(label,prob,th):

#     thresh, label = th, str(label)

#     print(label)

#     if ('c_' in label) & ('r_' in label) & ('v_' in label):

#         for s in ['c_','r_','v_']:

#             fo = label.index(s) + len(s)+1

#             label = label[:fo]+label[fo:]

#     elif 'c_' not in label:

#         p = prob[:7].numpy()

#         while all(p<thresh):

#             thresh -= 0.01

#         idx = np.argwhere(p>thresh)[0][0]

#         label = 'c_'+str(idx)+';'+label

#     elif 'r_' not in label:

#         p = prob[7:175].numpy()

#         while all(p<thresh):

#             thresh -= 0.01

#         idx = np.argwhere(p>thresh)[0][0]

#         label = label.split(';')[0]+';'+'r_'+str(idx)+';'+label.split(';')[1]

#     elif 'v_' not in label:

#         p = prob[175:].numpy()

#         while all(p<thresh):

#             thresh -= 0.01

#         idx = np.argwhere(p>thresh)[0][0]

#         label = label+';'+'v_'+str(idx)

#     label = [s.split('_')[1] for s in label.split(';')]

#     return label
%%time

row_id,target = [],[]

imgnet_mean = [0.485, 0.456, 0.406] # Here it's ImageNet statistics

imgnet_std = [0.229, 0.224, 0.225]





for fname in TEST:

    df = pd.read_parquet(fname)

    data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

    for idx in range(len(df)):

        name = df.iloc[idx,0]

        img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)

        img = crop_resize(img)

        img = np.stack((img,)*3,axis=-1)

        img = pil2tensor(img,np.float32).div_(255)

        pred, tsor, prob = learn.predict(Image(img))

        row_id += [f'{name}_consonant_diacritic',f'{name}_grapheme_root',f'{name}_vowel_diacritic']

        #labels = fillLabels(pred,prob,0.9)

        c = learn.data.classes

        #t = [c[i] for i in torch.topk(prob,3)[1].numpy()]

        target += [

                c[ np.argmax(prob[:7].numpy()) ].split('_')[1],

                c[ np.argmax(prob[7:175].numpy()) + 7 ].split('_')[1],

                c[ np.argmax(prob[175:].numpy()) + 175 ].split('_')[1] ]





sub_df = pd.DataFrame({'row_id': row_id, 'target': target})

sub_df.to_csv('submission.csv', index=False)

sub_df
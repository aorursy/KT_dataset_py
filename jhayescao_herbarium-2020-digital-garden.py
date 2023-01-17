import numpy as np

import pandas as pd

import os,random,time

from sklearn.model_selection import train_test_split

import torch

from torchvision import transforms

from PIL import Image

from torch.utils.data import Dataset,DataLoader

import torch.nn.functional as F

import json

import matplotlib.pyplot  as plt                   

import cv2
import json, codecs

with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/train/metadata.json", 'r',

                 encoding='utf-8', errors='ignore') as f:

    train_meta = json.load(f)

    

with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/test/metadata.json", 'r',

                 encoding='utf-8', errors='ignore') as f:

    test_meta = json.load(f)
train_meta.keys()
train_meta
sample_sub = pd.read_csv('../input/herbarium-2020-fgvc7/sample_submission.csv')

display(sample_sub)
train_df=pd.DataFrame(train_meta['annotations'])

train_df.head()
train_cat=pd.DataFrame(train_meta['categories'])

train_cat.columns = ['family', 'genus', 'category_id', 'category_name']

train_cat.head()
train_img=pd.DataFrame(train_meta['images'])

train_img.columns = ['file_name', 'height', 'image_id', 'license', 'width']

train_img.head()
train_reg = pd.DataFrame(train_meta['regions'])

train_reg.columns = ['region_id', 'region_name']

train_reg.head()
train_df = train_df.merge(train_cat, on='category_id', how='outer')

train_df = train_df.merge(train_img, on='image_id', how='outer')

train_df = train_df.merge(train_reg, on='region_id', how='outer')
train_df.head()
na = train_df.file_name.isna()

keep = [x for x in range(train_df.shape[0]) if not na[x]]

train_df = train_df.iloc[keep]
dtypes = ['int32', 'int32', 'int32', 'int32', 'object', 'object', 'object', 'object', 'int32', 'int32', 'int32', 'object']

for n, col in enumerate(train_df.columns):

    train_df[col] = train_df[col].astype(dtypes[n])

print(train_df.info())

display(train_df)
test_df = pd.DataFrame(test_meta['images'])

test_df.columns = ['file_name', 'height', 'image_id', 'license', 'width']

print(test_df.info())

display(test_df)
train_df.to_csv('full_train_data.csv', index=False)

test_df.to_csv('full_test_data.csv', index=False)
train_df.head()
test_df.head()
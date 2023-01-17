# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import cv2

import glob

import pydicom

import numpy as np

import pandas as pd

import torch

import torch.optim as optim

from albumentations import Compose, ShiftScaleRotate, Resize, CenterCrop, HorizontalFlip, RandomBrightnessContrast, Normalize

from albumentations.pytorch import ToTensor

from torch.utils.data import Dataset

from tqdm import tqdm_notebook as tqdm

from matplotlib import pyplot as plt

from torchvision import transforms
path = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'

path_trn = path+'stage_2_train'

path_tst = path+'stage_2_test'



path_dfs = '../input/rsnakerneldataframe/'

path_df_sample = path_dfs+'df_sample.fth'

path_df_trn = path_dfs+'df_trn.fth'

path_df_tst = path_dfs+'fns_tst.fth'

path_df_lbls = path_dfs+'labels.fth'

path_bins = path_dfs+'bins.pkl'
df_trn = pd.read_feather(path_df_trn)
df_lbls = pd.read_feather(path_df_lbls)
complete_df = df_trn.join(df_lbls.set_index('ID'), 'SOPInstanceUID')

assert not len(complete_df[complete_df['any'].isna()])
complete_df = complete_df.assign(pct_cut = pd.cut(complete_df.img_pct_window, [0,0.02,0.03,0.05,0.1,0.3,1]))

complete_df.drop(complete_df.query('img_pct_window<0.02').index, inplace=True)
df_lbl = complete_df.query('any==True')

len(df_lbl)
df_nonlbl = complete_df.query('any==False').sample(len(df_lbl))

len(df_nonlbl)
complete_df = pd.concat([df_lbl,df_nonlbl])

len(complete_df)

del(df_nonlbl, df_lbl)
df_samp_idx = complete_df.sample(frac=0.05)

df_samp_idx.to_csv('df_samp_idx.csv')

del(complete_df)
#img_file = os.path.join(path_trn, list(map(filename, df_samp.fname))[0]+'.dcm')

for index, row in df_samp_idx.iterrows():

    img_file = df_samp_idx.fname[index]

    dcm_name = img_file.rsplit("/")[-1].rsplit(".")[0]

    img_name = pydicom.dcmread(img_file)

    img_name = img_name.pixel_array

    cv2.imwrite(dcm_name + '.png', img_name) # write png image os.path.join(path , 'waka.jpg')
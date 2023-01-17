%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import os

import scipy as sp

import torch

from functools import partial

from sklearn import metrics

from collections import Counter

from fastai.callbacks import *



import PIL

import cv2
path = Path('../input/aptos2019-blindness-detection')
df = pd.read_csv(path/'train.csv')

df.head()
src = (

         ImageList.from_df(df, path, folder='train_images', suffix='.png')

        .split_by_rand_pct(0.2, seed=42)

        .label_from_df(cols='diagnosis', label_cls=FloatList)    

        )
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0.10, max_zoom=1.3, max_warp=0.0, max_lighting=0.2)
data = (

    src.transform(tfms,size=128)

    .databunch()

    .normalize(imagenet_stats)

)
# Definition of Quadratic Kappa

from sklearn.metrics import cohen_kappa_score

def quadratic_kappa(y_hat, y):

    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')
!mkdir -p /tmp/.cache/torch/checkpoints

!cp ../input/resnet50/resnet50.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth
learn = cnn_learner(data, models.resnet50, metrics=[quadratic_kappa], model_dir = Path('../kaggle/working'),

                   path = Path("."))
lr = 1e-3

learn.fit_one_cycle(4, lr)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.unfreeze()

lr = slice(1e-4, 1e-3)

learn.fit_one_cycle(6, max_lr=lr)
learn.fit_one_cycle(2, max_lr=lr)
data = (

    src.transform(tfms,size=224)

    .databunch()

    .normalize(imagenet_stats)

)
learn.data = data
lr = 1e-3

learn.fit_one_cycle(4, lr)
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 1e-5

learn.fit_one_cycle(4, max_lr=lr)
sample_df = pd.read_csv(path/"sample_submission.csv")

sample_df.head()
learn.data.add_test(ImageList.from_df(sample_df, path, folder='test_images', suffix='.png'))
preds,y = learn.get_preds(DatasetType.Test)
int_preds = [int(x) for x in preds]

sample_df.diagnosis = int_preds

sample_df.head()

sample_df.to_csv('submission.csv', index=False)
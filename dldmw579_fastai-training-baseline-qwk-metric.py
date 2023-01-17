import numpy as np 

import pandas as pd 

import os

from fastai import *

from fastai.vision import *

import pandas as pd

import matplotlib.pyplot as plt

import torch

from fastai.metrics import KappaScore
SIZE = 256

BS = 64
train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
train_df.head(1)
train = ImageList.from_df(train_df, path='../input/panda-challenge-resized-dataset/', cols='image_id',suffix='.jpeg')
src = (train

       .split_by_rand_pct(0.2) #split the dataset such that we have 20% as validation set

#        .split_from_df()

       .label_from_df(cols='isup_grade',label_cls=FloatList))
tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.1,max_lighting=0.1,p_lighting=0.5)
data= (src.transform(tfms,size=SIZE) #Data augmentation

       .databunch(bs=BS,num_workers=1) 

       .normalize(imagenet_stats))
data.show_batch(2)
from sklearn.metrics import cohen_kappa_score

def quadratic_kappa(y_hat, y):

    return torch.tensor(cohen_kappa_score(torch.round(y_hat).cpu(), y.cpu(), weights='quadratic'),device='cuda:0')
learn = cnn_learner(data,models.resnet18,metrics=[accuracy, quadratic_kappa], model_dir='../working/models/')

learn.path = Path('')
learn.fit_one_cycle(2,2e-3)

learn.save('resnet18')
learn.recorder.plot_losses()

learn.recorder.plot_metrics()
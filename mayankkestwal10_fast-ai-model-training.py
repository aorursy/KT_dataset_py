%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

from sklearn.metrics import cohen_kappa_score

import torch
Path = '../input/aptos2019-blindness-detection/'
train_df = pd.read_csv(Path+'train.csv')

test_df = pd.read_csv(Path+'test.csv')
train_df.head()
train_df['id_code'] = train_df['id_code'].apply(lambda x : x + '.png')

test_df['id_code'] = test_df['id_code'].apply(lambda x : x + '.png')
train_df.head()
bs = 32

SIZE = 224



tfms = get_transforms(do_flip=True,flip_vert=True,max_warp=0.,max_rotate=360.0)
data = (ImageList.from_df(df=train_df,folder='train_images',path=Path)

       .split_by_rand_pct(0.2)

       .label_from_df(cols='diagnosis')

       .transform(tfms,size=SIZE)

       .databunch(bs=bs)

       .normalize(imagenet_stats))
data
data.show_batch(rows=5,fig_size=(5,5))
!mkdir -p '/tmp/.cache/torch/checkpoints/'

!cp ../input/resnet101/resnet101.pth /tmp/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth
arch = models.resnet101
kappa = KappaScore()

kappa.weights = "quadratic"
learn = cnn_learner(data, arch, metrics=[kappa,error_rate,accuracy],pretrained=True,path='../working/')
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr = learn.recorder.min_grad_lr

lr
learn.fit_one_cycle(5, lr)
learn.recorder.plot_losses()
learn.save('stage1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr = learn.recorder.min_grad_lr

lr
learn.fit_one_cycle(4, max_lr=slice(lr,lr/10))
learn.save('stage2')
SIZE = 256



data = (ImageList.from_df(df=train_df,folder='train_images',path=Path)

       .split_by_rand_pct(0.2)

       .label_from_df(cols='diagnosis')

       .transform(tfms,size=SIZE)

       .databunch(bs=bs)

       .normalize(imagenet_stats))
data
learn.data = data
learn.freeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr = learn.recorder.min_grad_lr

lr
learn.fit_one_cycle(5, slice(lr,lr/10))
learn.save('stage3')
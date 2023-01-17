import numpy as np 

import pandas as pd 

import os

import torch

import torch.nn as nn

import matplotlib.pyplot as plt

%matplotlib inline

import fastai

from fastai.vision import *
path = untar_data(URLs.FLOWERS)
src = (ImageList.from_folder(path)

        .split_by_rand_pct(0.2, seed=42)

        .label_from_folder())
os.listdir(str(path))
with open(str(path)+'/train.txt', 'r') as f:

    c = f.read()
with open(str(path)+'/valid.txt', 'r') as f:

    v = f.read()
with open(str(path)+'/test.txt', 'r') as f:

    te = f.read()
lbf = c.split('\n')

lbfn = [f.split() for f in lbf]



lbfv = v.split('\n')

lbfvn = [f.split() for f in lbfv]



val_fn = [f[0] for f in lbfvn if len(f) > 0]



tlb = te.split('\n')

tlbfn = [f.split() for f in tlb]



test = pd.DataFrame()

test['img'] = tlbfn



train_list = lbfn+lbfvn



df = pd.DataFrame()

df['l'] = train_list

df['ll'] = df['l'].apply(len)

df = df[df['ll']>0].copy()

df['path'] = df['l'].apply(lambda x: x[0])

df['target'] = df['l'].apply(lambda x: x[1])

df = df[['path', 'target']].copy()



tlb = te.split('\n')

tlbfn = [f.split() for f in tlb]



test = pd.DataFrame()

test['l'] = tlbfn

test['ll'] = test['l'].apply(len)

test = test[test['ll']>0].copy()

test['path'] = test['l'].apply(lambda x: x[0])

test['target'] = test['l'].apply(lambda x: x[1])

test = test[['path', 'target']].copy()
df.shape, test.shape
test.head()
val_idx = df.loc[df.path.isin(val_fn)].index
len(val_idx)
open_image(path/df['path'].loc[1]).data.shape
tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=5.0, max_zoom=0.2,

                      max_lighting=0.05, max_warp=0.02)
data = (ImageList.from_df(df=df,cols='path',path=path)

        .split_by_rand_pct(0.2)

        .label_from_df('target')

        .transform(tfms, size=224)

        .databunch(bs=16,num_workers=2)

        .normalize(imagenet_stats)

       )
learn = cnn_learner(data,

                   models.resnet18,

                   metrics=[accuracy]).to_fp16()
learn.data.show_batch()
learn.fit_one_cycle(5, 1e-3)
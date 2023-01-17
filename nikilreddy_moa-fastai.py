from fastai import *

from fastai.tabular import *



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train_x=pd.read_csv('../input/lish-moa/train_features.csv')

train_y=pd.read_csv('../input/lish-moa/train_targets_scored.csv')



test_x=pd.read_csv('../input/lish-moa/test_features.csv')
print(train_x.shape)

train_x.head()
print(train_y.shape)

train_y.head()
for col in train_y.columns:

  if col!='sig_id':

    print(train_y[col].value_counts())
print(test_x.shape)

test_x.head()
train_x.info()
train_x.iloc[:,:10].info()
test_x.info()
train_y.info()
print('missing values in train_x:',sum(train_x.isnull().sum()))

print('missing values in test_x:',sum(test_x.isnull().sum()))

print('missing values in train_y:',sum(train_y.isnull().sum()))
print('train:')

print(train_x.cp_time.value_counts())

print(' ')

print(train_x.cp_type.value_counts())

print(' ')

print(train_x.cp_dose.value_counts())

print('--------------------------------------')

print('test:')

print(test_x.cp_time.value_counts())

print(' ')

print(test_x.cp_type.value_counts())

print(' ')

print(test_x.cp_dose.value_counts())
cat=['cp_time','cp_dose','cp_type']

cont=[x for x in train_x.columns if x not in cat and x!='sig_id']

depv=[x for x in train_y.columns if x!='sig_id']

print('categorical features:',len(cat))

print('continuous features:',len(cont))

print('dependent variables:',len(depv))
train=train_x

print('train_shape:',train.shape)
print(len(cat)+len(cont)+len(depv)+1)

train[depv]=train_y[depv]

print('train_shape:',train.shape)
train.head()
procs = [FillMissing,Categorify,Normalize]

import torch.nn as nn
procs = [FillMissing,Categorify,Normalize]

data=(TabularList.from_df(train, cont_names=cont, cat_names=cat, procs=procs)

                .split_by_rand_pct(valid_pct=0.15,seed=42)

                .label_from_df(cols=depv)

                .add_test(TabularList.from_df(test_x,procs=procs, cont_names=cont, cat_names=cat))

                .databunch())
learn=tabular_learner(data,layers=[1000,1000,1000])

learn.loss_func = nn.BCEWithLogitsLoss()
#learn
learn.fit_one_cycle(7, max_lr=slice(1e-4),wd=0.1)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(7, max_lr=slice(1e-5),wd=0.05)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(7, max_lr=slice(5e-6),wd=0.02)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
v=math.pow(10,-15)

def const1(a):

  for i in range(a.shape[0]):

    for j in range(a.shape[1]):

      if a[i][j]<v:

        a[i][j]=v

      elif a[i][j]>1-v:

        a[i][j]=1-v

  return a
const1(preds)
sub=pd.read_csv('../input/lish-moa/sample_submission.csv')

sub.head()
sub[depv]=preds

print(sub.shape)

sub.head()
sub.to_csv('submission.csv', header=True,index=False)
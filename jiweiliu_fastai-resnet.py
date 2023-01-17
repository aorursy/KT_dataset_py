import os

GPU_id = 0

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
import warnings

warnings.filterwarnings("ignore")



from fastai.vision import *

from fastai.train import Learner

from fastai.callbacks import SaveModelCallback

from fastai.metrics import accuracy as fastai_accuracy

from fastai.callbacks import SaveModelCallback

import torch.nn.functional as F

import torch



import pandas as pd

import numpy as np

import time
class MyImageList(ImageList):

    @classmethod

    def from_df(cls, df:DataFrame, cols:IntsOrStrs=0, **kwargs)->'ItemList':        

        res = super().from_df(df, path='./', cols=cols, **kwargs)  

        if 'label' in df.columns:

            res.items = df.drop('label',axis=1).values

        else:

            res.items = df.values

        res.c,res.sizes = 1,{}

        return res

        

    def get(self, i):

        res = torch.tensor(self.items[i].reshape([28,28])).float().unsqueeze(0)

        self.sizes[i] = res.size

        return Image(res)   
path = Path('../input/digit-recognizer')

path.ls()
train = pd.read_csv(path/'train.csv')

test = pd.read_csv(path/'test.csv')
il = MyImageList.from_df(train)

il
il[0].show(cmap='gray')
sd = il.split_by_rand_pct(0.2)

sd
ll = sd.label_from_df(cols='label')

ll
tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])
ll = ll.transform(tfms)
%%time

bs = 128

data = ll.databunch(bs=bs).normalize()

data.add_test(MyImageList.from_df(test))
def _plot(i,j,ax): data.train_ds[0][0].show(ax,cmap='gray')

plot_multi(_plot, 3, 3, figsize=(8,8))
xb,yb = data.one_batch()

print(xb.shape,yb.shape)

data.show_batch(rows=3, figsize=(10,8), cmap='gray')
class ResBlock(nn.Module):

    def __init__(self, nf):

        super().__init__()

        self.conv1 = conv_layer(nf,nf)

        self.conv2 = conv_layer(nf,nf)

        

    def forward(self, x): return x + self.conv2(self.conv1(x))

    

def conv2(ni,nf): return conv_layer(ni,nf,stride=2)    

def conv_and_res(ni,nf): return nn.Sequential(conv2(ni, nf), res_block(nf))
model = torch.nn.Sequential(

    conv_and_res(1, 8),

    conv_and_res(8, 16),

    conv_and_res(16, 32),

    conv_and_res(32, 16),

    conv2(16, 10),

    Flatten()

)
%%time

learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)

learn.model_dir = '/kaggle/working/models'
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(10,max_lr=slice(0.05),callbacks=[

            SaveModelCallback(learn, every='improvement', monitor='accuracy'),

            ])
%%time

yp,yr = learn.get_preds()

yp = yp.numpy()

yr = yr.numpy()
def cross_entropy(y,yp):

    # y is the ground truch

    # yp is the prediction

    yp[yp>0.99999] = 0.99999

    yp[yp<1e-5] = 1e-5

    return np.mean(-np.log(yp[range(yp.shape[0]),y.astype(int)]))



def accuracy(y,yp):

    return (y==np.argmax(yp,axis=1)).mean()



def softmax(score):

    score = np.asarray(score, dtype=float)

    score = np.exp(score-np.max(score))

    score = score/(np.sum(score, axis=1).reshape([score.shape[0],1]))#[:,np.newaxis]

    return score
%%time

acc = accuracy(yr,yp)

ce = cross_entropy(yr,yp)

print('Valid ACC: %.4f Cross Entropy:%4f'%(acc,ce))
%%time

yps,_ = learn.get_preds(DatasetType.Test)

yps = yps.numpy()
sub = pd.DataFrame()

sub['ImageId'] = np.arange(yps.shape[0])+1

sub['Label'] = np.argmax(yps,axis=1)

sub.head()
from datetime import datetime

clock = "{}".format(datetime.now()).replace(' ','-').replace(':','-').split('.')[0]

out = 'fastai_%s_acc_%.4f_ce_%.4f.csv'%(clock,acc,ce)

print(out)

sub.to_csv(out,index=False)
import pandas as pd

import matplotlib.pyplot as plt



import numpy as np

import os

from sklearn.metrics import f1_score



from fastai import *

from fastai.vision import *



import torch

import torch.nn as nn

import torchvision

import cv2



from tqdm import tqdm

from skmultilearn.model_selection import iterative_train_test_split

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer

import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score

%load_ext autoreload

%autoreload
from torchvision.models import *
model_path='.'

path='../input/histopathologic-cancer-detection/'

train_folder=f'{path}train'

test_folder=f'{path}test'

train_lbl=f'{path}train_labels.csv'

ORG_SIZE=96



bs=64

num_workers=None # Apprently 2 cpus per kaggle node, so 4 threads I think

sz=96
from pathlib import Path

test_fnames=[str(file) for file in Path(test_folder).iterdir()]

df_trn=pd.read_csv(train_lbl)
df_WSI=pd.read_csv('../input/histopathologiccancerwsi/patch_id_wsi.csv')
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0.0, max_zoom=.15,

                      max_lighting=0.1, max_warp=0.15)
df_notinWSI=df_trn.set_index('id').drop(df_WSI.id)
valWSI=df_WSI.groupby(by='wsi')['id'].count().sample(frac=0.23).index
trnWSI=[i[0] for i in df_WSI.groupby(by='wsi')['id'] if i[0] not in valWSI]
len(trnWSI),len(valWSI)
val_idx=np.hstack([df_WSI.groupby(by='wsi')['id'].indices[WSI] for WSI in valWSI])
val_idx=np.append(df_notinWSI.index.values,df_WSI.id[val_idx])

trn_idx=np.hstack([df_WSI.groupby(by='wsi')['id'].indices[WSI] for WSI in trnWSI])

trn_idx=df_WSI.id[trn_idx]
val_idx=df_trn.reset_index().set_index('id').loc[val_idx,'index'].values

trn_idx=df_trn.reset_index().set_index('id').loc[trn_idx,'index'].values
np.random.shuffle(val_idx)

np.random.shuffle(trn_idx)
src = (ImageList.from_df(df_trn, path=path, suffix='.tif',folder='train')                

                .split_by_idxs(trn_idx,val_idx)

                .label_from_df(label_delim=' '))

src.add_test(test_fnames);
data=ImageDataBunch.create_from_ll(src, ds_tfms=tfms, size=sz,bs=bs)

stats=data.batch_stats()        

data.normalize(stats);
def auc_score(y_pred,y_true,tens=True):

    score=roc_auc_score(y_true[:,1],torch.sigmoid(y_pred)[:,1])

    if tens:

        score=tensor(score)

    else:

        score=score

    return score







class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):

        super(FocalLoss, self).__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.logits = logits

        self.reduce = reduce

    def forward(self, inputs, targets):

        if self.logits:

            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)

        else:

            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)

        pt = torch.exp(-BCE_loss)

        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss



        if self.reduce:

            return torch.mean(F_loss)

        else:

            return F_loss

        



learn = create_cnn(

    data,

    densenet169,

    path='.',    

    metrics=[auc_score], 

    #loss_func=FocalLoss(logits=True,gamma=1),

    ps=0.5

)
x,y=learn.get_preds()
auc_score(x,y)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1,1e-2)

learn.recorder.plot()

learn.recorder.plot_losses()
learn.unfreeze()

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(10,slice(1e-4,1e-3))
learn.recorder.plot()
learn.recorder.plot_losses()
preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn, preds, y.long(), losses)
preds,y=learn.get_preds()

pred_score=auc_score(preds,y)

pred_score
preds,y=learn.TTA()

pred_score_tta=auc_score(preds,y)

pred_score_tta
preds_test,y_test=learn.get_preds(ds_type=DatasetType.Test)
preds_test_tta,y_test_tta=learn.TTA(ds_type=DatasetType.Test)
sub=pd.read_csv(f'{path}/sample_submission.csv').set_index('id')

sub.head()
clean_fname=np.vectorize(lambda fname: str(fname).split('/')[-1].split('.')[0])

fname_cleaned=clean_fname(data.test_ds.items)

fname_cleaned=fname_cleaned.astype(str)
sub.loc[fname_cleaned,'label']=to_np(preds_test[:,1])

sub.to_csv(f'submission_{pred_score}.csv')
sub.loc[fname_cleaned,'label']=to_np(preds_test_tta[:,1])

sub.to_csv(f'submission_{pred_score_tta}.csv')
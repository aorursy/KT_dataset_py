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



%load_ext autoreload

%autoreload
model_path='.'

path='../input/'

train_folder=f'{path}train'

test_folder=f'{path}test'

train_lbl=f'{path}train_labels.csv'

ORG_SIZE=96



bs=64

num_workers=None # Apprently 2 cpus per kaggle node, so 4 threads I think

sz=96
df_trn=pd.read_csv(train_lbl)
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=.0, max_zoom=.1,

                      max_lighting=0.05, max_warp=0.)
data = ImageDataBunch.from_csv(path,csv_labels=train_lbl,folder='train', ds_tfms=tfms, size=sz, suffix='.tif',test=test_folder,bs=bs);

stats=data.batch_stats()        

data.normalize(stats)
data.show_batch(rows=5, figsize=(12,9))
from sklearn.metrics import roc_auc_score
def auc_score(y_pred,y_true,tens=True):

    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])

    if tens:

        score=tensor(score)

    else:

        score=score

    return score
from torchvision.models import *
learn = create_cnn(

    data,

    densenet201,

    path='.',    

    metrics=[auc_score], 

    ps=0.5

)
#print(learn.summary())
learn.lr_find()

learn.recorder.plot()
lr = 1e-04
learn.fit_one_cycle(1,lr)

learn.recorder.plot()

learn.recorder.plot_losses()
learn.unfreeze()

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1,slice(1e-4,1e-3)) #10
learn.recorder.plot()
learn.recorder.plot_losses()
m = learn.model.eval()
idx = 4
x,y = data.valid_ds[idx]

x.show()

data.valid_ds.y[idx]
xb,_ = data.one_item(x) #takes all the settings from our previously created data object

xb_im = Image(data.denorm(xb)[0])

xb = xb.cuda()
from fastai.callbacks.hooks import *
def hooked_backward(cat=y):

    with hook_output(m[0]) as hook_a: #Get activations from the convolution layers

        with hook_output(m[0], grad = True) as hook_g: #Get gradient from convolution layers

            preds = m(xb) #DO foreward pass through model

            preds[0,int(cat)].backward()

    return hook_a, hook_g

hook_a, hook_g = hooked_backward()
#hook_a.stored
acts = hook_a.stored[0].cpu()

acts.shape #Now we see our 512 chanels over the 11x11 sections of the image
avg_acts = acts.mean(0)

avg_acts.shape
def show_heatmap(hm):

    _,ax = plt.subplots()

    xb_im.show(ax) #fastai function to show the image

    ax.imshow(hm, alpha = 0.6, extent = (0,92,92, 0), #extent expands the 11x11 image to 352,352

             interpolation = 'bilinear', cmap = 'magma')
show_heatmap(avg_acts)
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
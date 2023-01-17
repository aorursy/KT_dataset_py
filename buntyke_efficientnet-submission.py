%reload_ext autoreload

%autoreload 2

%matplotlib inline



import re

import json

import math

import collections

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import style

from joblib import load, dump

from functools import partial

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline  



from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import StratifiedKFold

from fastai import *

from fastai.vision import *

from fastai.callbacks import *



import torch

from torch import nn

from torch.utils import model_zoo

from torch.nn import functional as F

from torchvision import models as md
import sys

package_dir = '../input/efficientnet/efficientnet_pytorch'

sys.path.insert(0, package_dir)



from efficientnet_pytorch import EfficientNet
# making model

md_ef = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1)
#copying weighst to the local directory 

!mkdir models
def get_df():

    base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')

    train_dir = os.path.join(base_image_dir,'train_images/')

    df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))

    df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))

    df = df.drop(columns=['id_code'])

    df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe

    test_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

    return df, test_df



df, test_df = get_df()
#you can play around with tfms and image sizes

bs = 64

sz = 224

tfms = get_transforms(do_flip=True,flip_vert=True)
data = (ImageList.from_df(df=df,path='./',cols='path') 

        .split_by_rand_pct(0.2) 

        .label_from_df(cols='diagnosis',label_cls=FloatList) 

        .transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 

        .databunch(bs=bs,num_workers=4) 

        .normalize(imagenet_stats)  

       )
def qk(y_pred, y):

    return torch.tensor(cohen_kappa_score(torch.round(y_pred), y, weights='quadratic'), device='cuda:0')
learn = Learner(data, 

                md_ef, 

                metrics = [qk], 

                model_dir="models").to_fp16()



learn.data.add_test(ImageList.from_df(test_df,

                                      '../input/aptos2019-blindness-detection',

                                      folder='test_images',

                                      suffix='.png'))
learn.fit_one_cycle(10,1e-3)
#https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa

class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

        print(-loss_partial(self.coef_['x']))



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']
def run_subm(learn=learn, coefficients=[0.5, 1.5, 2.5, 3.5]):

    opt = OptimizedRounder()

    preds,y = learn.get_preds(DatasetType.Test)

    tst_pred = opt.predict(preds, coefficients)

    test_df.diagnosis = tst_pred.astype(int)

    test_df.to_csv('submission.csv',index=False)

    print ('done')
run_subm()
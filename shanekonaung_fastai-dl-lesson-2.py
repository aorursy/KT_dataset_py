%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print(os.listdir('../input/'))
PATH = '../input/'
train = 'train-jpg'
labels = f'{PATH}train_v2.csv'
arch = resnet34
sz = 229
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"

l = pd.read_csv('../input/train_v2.csv')
l.describe()
val_idxs = get_cv_idxs(40479)
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_top_down, max_zoom=1.1)
data = ImageClassifierData.from_csv(PATH, folder=train, csv_fname=labels, tfms=tfms, 
                                    val_idxs=val_idxs, suffix='.jpg')
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.fit(1e-2, 1)








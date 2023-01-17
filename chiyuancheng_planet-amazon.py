%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

#from fastai.conv_learner import *



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

from spectral import *

import seaborn as sns





from fastai.imports import *

from sklearn.metrics import fbeta_score

import warnings



import torch

import torchvision

import torch.utils.data as data



pal = sns.color_palette()

sns.set_style("whitegrid")
path = Path('/kaggle/input/')

path.ls()
print('# File sizes')

for f in os.listdir('../input'):

    if not os.path.isdir('../input/' + f):

        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')

    else:

        sizes = [os.path.getsize('../input/'+f+'/'+x)/1000000 for x in os.listdir('../input/' + f)]

        print(f.ljust(30) + str(round(sum(sizes), 2)) + 'MB' + ' ({} files)'.format(len(sizes)))


df = pd.read_csv(path/'train_v2.csv')

df.head()
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
np.random.seed(42)

src = (ImageList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')

       .random_split_by_pct(0.2)

       .label_from_df(label_delim=' '))
data = (src.transform(tfms, size=128)

        .databunch(num_workers=0).normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(12,9))
arch = models.resnet50
acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)

learn = create_cnn(data, arch, metrics=[acc_02, f_score], model_dir='/tmp/models')
learn.lr_find()
learn.recorder.plot()
lr = 0.01
learn.fit_one_cycle(5, slice(lr))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
train='../input/dogs-vs-cats-redux-kernels-edition/train'
test='../input/dogs-vs-cats-redux-kernels-edition/test'
data = []
for files in sorted(os.listdir(train)):
    data.append((files))

df = pd.DataFrame(data, columns=['img'])
df['tag']=df['img'].astype(str).str[:3]
df['tag'].value_counts()
!mkdir ./data
df.to_csv('./data/catdog.csv',index=False)
label_csv = './data/catdog.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)
import gc
gc.collect()
PATH=''
arch=resnet34
sz=224
data = ImageClassifierData.from_csv(PATH,train,label_csv,tfms=tfms_from_model(arch, sz),test_name=test)

#import pathlib
#data.path = pathlib.Path('.')
from os.path import expanduser, join, exists
from os import makedirs
cache_dir = expanduser(join('~', '.torch'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)
!ls ~/.torch   #Now we have a .torch/models directory in the root folder 
#Lets copy the weights to the folder

!cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth

#Change the true path of the model
import pathlib
data.Path = pathlib.Path('.')
#learn = ConvLearner.pretrained(arch, data, precompute=True)
#learn.fit(0.01, 2)
#gc.collect()
#Two more epochs
#learn.fit(0.01, 2)
#gc.collect()
#log_preds, y = learn.TTA(is_test=True)
#probs = np.mean(np.exp(log_preds),0)
#ds=pd.DataFrame(probs)
#ds.columns=data.classes
#ds.insert(0,'id',[o.rsplit('/', 1)[1] for o in data.test_ds.fnames])
#subm=pd.DataFrame()
#subm['id']=ds['id']
#subm['cat']=ds['cat']
#subm=pd.DataFrame()
#subm['id']=ds['id']
#subm['cat']=ds['cat']
#subm.to_csv('../workingsubmission.csv',index=False)
#subm.to_csv('./data/submission.gz',compression='gzip',index=False)
#!ls ./data
#subm

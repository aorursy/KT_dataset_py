%reload_ext autoreload

%autoreload 2

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *

from fastai import *



import path



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input/flowers-recognition/flowers/flowers"))



# Any results you write to the current directory are saved as output.
#copy the fastai weights to kaggle, don't forget to toggle the internet on!

#!mkdir -p /tmp/.cache/torch/checkpoints

#!cp ../input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth

path = Path('../input/flowers-recognition/flowers/flowers')

path.ls()
classes = ['sunflower','tulip','daisy','rose','dandelion']
#GPU available

torch.cuda.is_available()
#has to be true

torch.backends.cudnn.enabled
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4)

data.normalize(imagenet_stats)
data.classes
data.show_batch(rows=3,figsize=(7,8))
data.classes , data.c , len(data.train_ds), len(data.valid_ds)
#Now we have to make a dir 

#for copying those weights and hence we are making ~/.torch/models folder. 

cache_dir = os.path.expanduser(os.path.join('~','.torch'))

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir) #first make ~/.torch if not already available.
models_dir = os.path.join(cache_dir,'models')

if not os.path.exists(models_dir):

    os.makedirs(models_dir) #then make ~/.torch/models, if not already available. 
#Copied resnet34.pth, which are pretrained weights on Resnet34 to our folder into resnet<version>-<sha-hash>.pth

!cp ../input/resnet34/resnet34.pth ~/.torch/models/resnet34-333f7ec4.pth 
learn = cnn_learner(data, models.resnet34, metrics=[error_rate,accuracy] ,model_dir= '/tmp/models/')
learn.fit_one_cycle(4)
#Saving the model with ACCURACY = 91%

learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9,figsize =(9,9))
interp.plot_confusion_matrix()
#Initiating refit and checking LR

learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
#Saving the model with ACCURACY = 92%

learn.save('stage-2')


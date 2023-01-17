# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input/"))
%reload_ext autoreload
%autoreload 2
%matplotlib inline
# This file contains all the main external libs we'll use
from fastai import *
from fastai.vision import *


PATH = "../input/fruits/fruits-360_dataset/fruits-360"
sz=299
bs=16
tfms = get_transforms(flip_vert=True,max_rotate=90.)
data = ImageDataBunch.from_folder(PATH, train="Training", valid="Test", 
    ds_tfms=tfms, size=sz,bs=bs, num_workers=0)#.normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
def get_ex(): return open_image(PATH+'/Training/Peach/r_295_100.jpg')

def plots_f(rows, cols, width, height, **kwargs):
    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(
        rows,cols,figsize=(width,height))[1].flatten())]
plots_f(4, 4, 12, 6, size=sz)
from os.path import expanduser, join, exists
from os import makedirs
cache_dir = expanduser(join('~', '.torch'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)

# copy time!
!cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth
learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir='/output/model/',callback_fns=ShowGraph)
lrf=learn.lr_find()
learn.recorder.plot()
lr=1e-2
learn.fit_one_cycle(1,lr)
learn.save('fruit_resnet34-stage-1')
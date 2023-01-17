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
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

from fastai.utils.collect_env import *
show_install(True)
np.random.seed(42)
path = Path('../input/fruits-360_dataset/fruits-360')
!ls {path}
train_path = path/'Training'

test_path = path/'Test'



ls_trn = train_path.ls()

ls_tst = test_path.ls()

len(ls_trn), len(ls_tst), ls_trn[90:], ls_tst[90:]
def add_test_folder(iil, test_path):

    iil.test = ImageItemList.from_folder(test_path).no_split().label_from_folder().train

    

iil = (ImageItemList.from_folder(train_path)

                     .random_split_by_pct(.2)

                     .label_from_folder())



add_test_folder(iil, test_path)



data = iil.transform(tfms=None, size=100, bs=32).databunch().normalize(imagenet_stats)
len(data.train_dl.dataset), len(data.valid_dl.dataset), len(data.test_dl.dataset)
data.show_batch(ds_type=DatasetType.Train ,rows=3, figsize=(7,7))
print(data.classes)

len(data.classes),data.c
metrics = [accuracy]
!pwd
learn = create_cnn(data, models.resnet34, model_dir='/kaggle/working/models',  metrics=metrics)
learn.loss_func
learn.lr_find()
learn.recorder.plot()
lr = 1e-3
learn.fit_one_cycle(5, max_lr=lr)
learn.recorder.plot_losses()
learn.save('fruits-stg1-rn34')

!ls -l ./models
results = learn.validate(dl=learn.data.test_dl)
print('loss: {:.6f}; accuracy: {}'.format(results[0].item(), results[1].item()))
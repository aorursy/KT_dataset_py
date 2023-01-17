# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# path = "../input/"

%matplotlib inline



# Any results you write to the current directory are saved as output

!pip install fastai -U

# !pip install fastai==1.0.46 --force-reinstall

!pip list

import torch

import fastai



from fastai import *

from fastai.vision import *



print(torch.__version__)

print(fastai.__version__)



print(torch.cuda.is_available())

print(torch.backends.cudnn.enabled)
!pip install torch -U
!pip install torchvision -U
path = untar_data(URLs.DOGS)

path
data = ImageDataBunch.from_folder(path, bs=16, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

data.show_batch(rows=3)
len(data.train_ds)
len(data.valid_ds)
learner = create_cnn(data, models.resnet34, metrics=accuracy).to_fp16()
learner.fit_one_cycle(1)
learner.unfreeze()

learner.fit_one_cycle(1, slice(1e-5,3e-4), pct_start=0.05)
learner.to_fp32()
accuracy(*learner.TTA())
preds, y, losses = learner.get_preds(with_loss=True)
interp = ClassificationInterpretation(learner, preds, y, losses)

interp.most_confused()
interp.plot_confusion_matrix()
interp.plot_top_losses(9, figsize=(7,7))
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
!pip install pytorchcv
import numpy as np 

import pandas as pd 

from pytorchcv.model_provider import get_model as ptcv_get_model

from pathlib import Path

from fastai import *

from fastai.vision import *

import torch
data_folder = Path("../input")

data_folder.ls()
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/sample_submission.csv")
test_img = ImageList.from_df(test_df, path='../input/test/test')

trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)

train_img = (ImageList.from_df(train_df, path='../input/train/train')

        .split_by_rand_pct(0.01)

        .label_from_df()

        .add_test(test_img)

        .transform(trfm, size=128)

        .databunch(path='.', bs=64, device= torch.device('cuda:0'))

        .normalize(imagenet_stats)

       )
learn = cnn_learner(train_img, models.resnet152, metrics=[error_rate, accuracy])
learn.lr_find()

learn.recorder.plot()
lr = 2e-02

learn.fit_one_cycle(4, slice(lr))
preds,_ = learn.get_preds(ds_type=DatasetType.Test)
test_df.has_cactus = preds.numpy()[:, 0]
test_df.to_csv('samplesubmission.csv', index=False)
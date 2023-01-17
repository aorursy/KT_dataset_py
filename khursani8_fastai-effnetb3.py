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
!pip install efficientnet_pytorch
from fastai.vision import *

path = Path('../input/')
path.ls()
train = pd.read_csv(path/'train.csv')

test = pd.read_csv(path/'sample_submission.csv')
test_img = ImageList.from_df(test, path=path/'test', folder='test')

tfms = get_transforms()

data = (ImageList.from_df(train, path=path/'train', folder='train')

        .split_by_rand_pct(0.01)

        .label_from_df()

        .add_test(test_img)

        .transform(tfms, size=224, resize_method=ResizeMethod.SQUISH)

        .databunch(path='.', bs=64, device= torch.device('cuda:0'))

        .normalize(imagenet_stats)

       )
from efficientnet_pytorch import EfficientNet
model_name = 'efficientnet-b3'

def getModel(pret):

    model = EfficientNet.from_pretrained(model_name)

    model._fc = nn.Linear(model._fc.in_features,data.c)

    return model
learn = Learner(data,getModel(True),metrics=[accuracy])
# learn.lr_find()

# learn.recorder.plot()
lr=5e-3
learn.fit_one_cycle(3,lr)
preds,_ = learn.get_preds(ds_type=DatasetType.Test)

# preds,_ = learn.TTA(ds_type=DatasetType.Test)

idx = preds.numpy()[:,0]
test.has_cactus = idx

test.to_csv('submission.csv', index=False)
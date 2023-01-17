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
import os

import json

from fastai.vision import *

from fastai.callbacks import *

import torch.nn as nn
print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled, torch.backends.cudnn.deterministic)

torch.backends.cudnn.deterministic = True

print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled, torch.backends.cudnn.deterministic)
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0, max_warp=0)

tfms
data_path = "../input/aptos2019-blindness-detection"

train_label_file = "train.csv"

train_images_folder = "train_images"

test_label_file = "test.csv"

test_images_folder = "test_images"

image_suffix = ".png"
split_pct = 0.1

bs = 64

size = 224
fname = os.path.join(data_path, train_label_file)
df = pd.read_csv(fname)

df.head(5)
_ = df.hist()
np.random.seed(42)

src = (ImageList.from_csv(data_path, train_label_file, folder=train_images_folder, suffix=image_suffix)

       .split_by_rand_pct(split_pct)

       .label_from_df())

src
data = (src.transform(tfms, size=size)

        .databunch(bs=bs).normalize())

data.c, data.classes
data.show_batch(rows=3, figsize=(7, 6))
arch = models.vgg16_bn
kappa = KappaScore()

kappa.weights = "quadratic"
if not os.path.exists('/tmp/.cache/torch/checkpoints/'):

        os.makedirs('/tmp/.cache/torch/checkpoints/')
print(os.listdir("/tmp/.cache/torch/checkpoints"))
! cp "../input/fastai-pretrained-models/vgg16_bn-6c64b313.pth" "/tmp/.cache/torch/checkpoints/vgg16_bn-6c64b313.pth"
learn = cnn_learner(data, arch, metrics=[error_rate, accuracy, kappa], model_dir= "../saved_models")
learn.loss_func
class FocalLoss(nn.Module):

    def __init__(self, gamma=3., reduction='mean'):

        super().__init__()

        self.gamma = gamma

        self.reduction = reduction



    def forward(self, inputs, targets):

        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

        pt = torch.exp(-CE_loss)

        F_loss = ((1 - pt)**self.gamma) * CE_loss

        if self.reduction == 'sum':

            return F_loss.sum()

        elif self.reduction == 'mean':

            return F_loss.mean()
learn.loss_func = FocalLoss()

learn.loss_func
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 1e-2/2
learn.fit_one_cycle(5, lr,  callbacks=[SaveModelCallback(learn, monitor='kappa_score', mode='max', name="headonly_vgg16_bn")])
learn.recorder.plot_losses()

learn.recorder.plot_metrics()
learn.load('headonly_vgg16_bn');
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2, slice(1e-4, lr/5), callbacks=[SaveModelCallback(learn, monitor='kappa_score', mode='max', name="best_entirenet_vgg16_bn_224")])
learn.recorder.plot_losses()

learn.recorder.plot_metrics()
interp = ClassificationInterpretation.from_learner(learn)
learn.load('best_entirenet_vgg16_bn_224');
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample_df.head()
test = ImageList.from_df(sample_df, data_path, folder=test_images_folder, suffix=image_suffix)

test
learn.data.add_test(test, label=test_label_file)
preds, y = learn.get_preds(ds_type=DatasetType.Test)
sample_df.diagnosis = preds.argmax(1)

sample_df.head()
sample_df.to_csv('submission.csv',index=False)

_ = sample_df.hist()
print(os.listdir("../input/saved_models"))
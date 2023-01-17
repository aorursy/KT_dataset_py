

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai import *

from fastai.vision import *

import torch

from pathlib import *
path=Path("../input")

path.ls()
train=pd.read_csv('../input/train.csv')

train.head()
train.has_cactus.value_counts().plot(kind='bar')
test=pd.read_csv('../input/sample_submission.csv')
test.head()
test_img=ImageList.from_df(test, path=path/"test", folder="test")

tfms=get_transforms(flip_vert=True)
train_img= (ImageList.from_df(train, path=path/'train', folder="train")

            .split_by_rand_pct(0.01)

            .label_from_df()

            .add_test(test_img)

            .transform(tfms, size=256)

            .databunch(path='../input', bs=64)

            .normalize(imagenet_stats)

           )
train_img.show_batch(rows=3)
learn=cnn_learner(train_img, models.resnet50, metrics=[error_rate, accuracy], path=".")
lr = 3e-3

learn.fit_one_cycle(3, lr)
probability, classification = learn.get_preds(ds_type=DatasetType.Test)

test.has_cactus = probability.numpy()[:, 0]

test.head()
test.to_csv("submission_2.csv", index=False)
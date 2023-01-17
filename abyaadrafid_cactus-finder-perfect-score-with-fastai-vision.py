import numpy as np

import pandas as pd

from pathlib import Path

from fastai import *

from fastai.vision import *

import torch

data_folder = Path("../input")

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/sample_submission.csv")
test_img = ImageList.from_df(test_df, path=data_folder/'test', folder='test')
src = (ImageList.from_df(train_df, path=data_folder/'train', folder='train')

        .split_by_rand_pct(0.01)

        .label_from_df()

        .add_test(test_img)

       )
train_img=src.databunch('.',bs=50)
train_img.show_batch()
tfms=get_transforms(flip_vert=True)
train_img = (src.transform(tfms,size=128)

            .databunch('.',bs=50)

       )
denselearner = cnn_learner(train_img, models.densenet161, metrics=[FBeta(),error_rate, accuracy])
denselearner.lr_find()

denselearner.recorder.plot(suggestion=True)
lr = 7.5e-03

denselearner.fit_one_cycle(5, slice(lr))
denselearner.unfreeze()

denselearner.lr_find()

denselearner.recorder.plot(suggestion=True)
denselearner.fit_one_cycle(1, slice(1e-06))
reslearner = cnn_learner(train_img, models.resnet101, metrics=[FBeta(),error_rate, accuracy])
reslearner.lr_find()
reslearner.recorder.plot(suggestion=True)
lr=9e-3
reslearner.fit_one_cycle(5,slice(lr))
reslearner.unfreeze()

reslearner.fit_one_cycle(2,slice(1e-6))
interp = ClassificationInterpretation.from_learner(reslearner)

interp.plot_top_losses(9, figsize=(7,6))
preds,_ = reslearner.get_preds(ds_type=DatasetType.Test)
test_df.has_cactus = preds.numpy()[:, 0]
test_df.to_csv('submission.csv', index=False)
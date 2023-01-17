import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path

from fastai import *

from fastai.vision import *

#import torch
data_folder = Path("")

#data_folder.ls()
train_df = pd.read_csv("../input/aerial-cactus-identification/train.csv")

#train_df.head()

test_df = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")

#test_df.head()
test_img = ImageList.from_df(test_df, path='../input/aerial-cactus-identification/test/test')
trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
train_img = (ImageList.from_df(train_df, path='../input/aerial-cactus-identification/train/train')

        .split_by_rand_pct(0.01)

        .label_from_df()

        .add_test(test_img)

        .transform(trfm, size=128)

        .databunch(path='.', bs=64, device= torch.device('cuda:0'))

        .normalize(imagenet_stats)

       )
train_img.show_batch(rows=3, figsize=(7,6))
#ttes_img.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(train_img, models.densenet161, metrics=[error_rate, accuracy])
learn.lr_find()

learn.recorder.plot()
lr = 3e-02

learn.fit_one_cycle(5, slice(lr))
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, slice(1e-06))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(7,6))
preds,_ = learn.get_preds(ds_type=DatasetType.Test)
test_df.has_cactus = preds.numpy()[:, 0]
test_df.to_csv('submission.csv', index=False)
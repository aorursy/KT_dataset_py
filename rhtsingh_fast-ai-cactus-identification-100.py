import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')

%config InlineBackend.figure_format = "retina"

import seaborn as sns

from IPython.display import display

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%load_ext autoreload

%autoreload 2

%precision 2
from pathlib import Path

from fastai import *

from fastai.vision import *

import torch
data_folder = Path("../input/")

data_folder.ls()
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/sample_submission.csv')
def display_all(df):

    with pd.option_context("max_rows",100,"max_columns",df.shape[0]):

        display(df)
display_all(train_df)
test_img = ImageList.from_df(test_df,path=data_folder/'test',folder='test')

trfm = get_transforms(do_flip=True,flip_vert=True,max_rotate=10.0,max_zoom=1.1, \

                      max_lighting=0.2,max_warp=0.2,p_affine=0.75,p_lighting=0.75)
train_img = (ImageList.from_df(train_df, path=data_folder/'train', folder='train')

        .split_by_rand_pct(0.01)

        .label_from_df()

        .add_test(test_img)

        .transform(trfm, size=128)

        .databunch(path='.', bs=64, device= torch.device('cuda:0'))

        .normalize(imagenet_stats)

       )
train_img.show_batch(rows=5, figsize=(12,12))
learn = cnn_learner(train_img, models.densenet161, metrics=[error_rate, accuracy])
learn.lr_find()
learn.recorder.plot()
lr = 3e-02

learn.fit_one_cycle(5,slice(lr))
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, slice(1e-06))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(7,6))
preds,_ = learn.get_preds(ds_type=DatasetType.Test)
test_df.has_cactus = preds.numpy()[:, 0]
test_df.to_csv('submission.csv', index=False)
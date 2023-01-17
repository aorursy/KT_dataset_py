import numpy as np 

import pandas as pd

from pathlib import Path

from fastai import *

from fastai.vision import *

import torchvision

import torch
data_root_path = Path("../input")
train_df = pd.read_csv(data_root_path/"train_labels.csv")

test_df = pd.read_csv(data_root_path/"sample_submission.csv")

train_df.id = train_df.id + '.tif'

test_df.id = test_df.id + '.tif'
transforms = get_transforms(

    do_flip=True, 

    flip_vert=True, 

    max_rotate=15.0, 

    max_lighting=0.2, 

    max_warp=0.2

)



train_imgs = ImageList.from_df(train_df, path=data_root_path, folder='train')

test_imgs = ImageList.from_df(test_df, path=data_root_path, folder='test')



train_imgs = (train_imgs

    .split_by_rand_pct(0.01)

    .label_from_df()

    .add_test(test_imgs)

    .transform(transforms, size=128)

    .databunch(path='.', bs=64, device= torch.device('cuda:0'))

    .normalize(imagenet_stats))
learn = cnn_learner(train_imgs, torchvision.models.densenet169, metrics=[error_rate, accuracy])
learn.lr_find()

learn.recorder.plot()
lr = 3e-02

learn.fit_one_cycle(10, slice(lr))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(7,6))
preds,_ = learn.get_preds(ds_type=DatasetType.Test)
test_df.label = preds.numpy()[:, 0]
test_df['id'] = test_df['id'].str.replace('.tif','')
test_df.to_csv('submission.csv', index=False)
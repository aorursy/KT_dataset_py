from fastai.vision import *

from fastai import *

import os

import pandas as pd

import numpy as np

print(os.listdir("../input/"))
train_dir="../input/train/train"

test_dir="../input/test/test"

train = pd.read_csv('../input/train.csv')

test = pd.read_csv("../input/sample_submission.csv")

data_folder = Path("../input")

train.head(5)

train.describe()
test_img = ImageList.from_df(test, path=data_folder/'test', folder='test')

# Applying Data augmentation

trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)

train_img = (ImageList.from_df(train, path=data_folder/'train', folder='train')

        .split_by_rand_pct(0.01)

        .label_from_df()

        .add_test(test_img)

        .transform(trfm, size=128)

        .databunch(path='.', bs=64, device= torch.device('cuda:0'))

        .normalize(imagenet_stats)

       )
learn = cnn_learner(train_img, models.densenet161, metrics=[error_rate, accuracy])

learn.lr_find()



learn.recorder.plot()
lr = 1e-02

learn.fit_one_cycle(3, slice(lr))



preds,_ = learn.get_preds(ds_type=DatasetType.Test)
test.has_cactus = preds.numpy()[:, 0]
test.to_csv('submission.csv', index=False)
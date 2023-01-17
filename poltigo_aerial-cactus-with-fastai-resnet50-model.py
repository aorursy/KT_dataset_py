from fastai.vision import *

import pandas as pd

import numpy as np
data = Path("../input/aerial-cactus-identification")

data.ls()
train_df = pd.read_csv("../input/aerial-cactus-identification/train.csv")

test_df = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")
test_img = ImageList.from_df(test_df, path=data/'test', folder='test')

trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)

train_img = (ImageList.from_df(train_df, path=data/'train', folder='train')

        .split_by_rand_pct(0.01)

        .label_from_df()

        .add_test(test_img)

        .transform(trfm, size=128)

        .databunch(path='.', bs=64, device= torch.device('cuda:0'))

        .normalize(imagenet_stats)

       )

train_img.show_batch(rows=3, figsize=(7,6))

learn = cnn_learner(train_img, models.resnet50, metrics=[error_rate, accuracy])
learn.fit_one_cycle(5)
learn.unfreeze()
learn.lr_find(start_lr=1e-5, end_lr=1e-1)

learn.recorder.plot()
lr = 1e-04
learn.fit_one_cycle(2, max_lr=lr)
probability, classification = learn.get_preds(ds_type=DatasetType.Test)

test_df.has_cactus = probability.numpy()[:, 0]

test_df.head()
test_df.to_csv("submission.csv", index=False)
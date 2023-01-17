import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai import *

from fastai.vision import *
data_folder = Path("../input/aerial-cactus-identification")

train_df = pd.read_csv("../input/aerial-cactus-identification/train.csv")

test_df = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")

test_img = ImageList.from_df(test_df, path=data_folder/'test', folder='test')

trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)

train_img = (ImageList.from_df(train_df, path=data_folder/'train', folder='train')

        .split_by_rand_pct(0.01)

        .label_from_df()

        .add_test(test_img)

        .transform(trfm, size=128)

        .databunch(bs=64, device= torch.device('cuda:0'))

        .normalize(imagenet_stats)

       )
# train_img.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(train_img, models.resnet18, metrics=[error_rate, accuracy])

# Vì lý do read only mà không tìm lr được :D ?!?

# learn.lr_find()

# learn.recorder.plot()
lr = 3e-02

learn.fit_one_cycle(5, slice(lr))
preds,_ = learn.get_preds(ds_type=DatasetType.Test)

test_df.has_cactus = preds.numpy()[:, 0]

test_df.to_csv('submission.csv', index=False)
from IPython.display import FileLink

FileLink('submission.csv')
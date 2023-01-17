# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from fastai.vision import *
import fastai

defaults.device = torch.device('cuda')
!unzip /kaggle/input/aerial-cactus-identification/train.zip
!unzip /kaggle/input/aerial-cactus-identification/test.zip
train_df = pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')
train_df.head()
tmp_df = pd.read_csv('/kaggle/input/aerial-cactus-identification/sample_submission.csv')
tmp_df.head()
test_ds = ImageList.from_df(tmp_df, path='.', folder='test')
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate= 15.0, 
                      max_zoom=1.1, max_lighting=0.15, max_warp=0.15,
                      p_affine=0.75, p_lighting=0.75)
train_ds1 = (ImageList.from_df(train_df, path='.', folder='train')
                            .split_by_rand_pct(0.15)
                            .label_from_df()
                            .add_test(test_ds)
                            .transform(tfms, size=128)
                            .databunch(path='.', bs=128)
                            .normalize(imagenet_stats))
train_ds1.show_batch(rows=4, figsize=(6,6))
train_ds2 = (ImageList.from_df(train_df, path='.', folder='train')
                            .split_by_rand_pct(0.1)
                            .label_from_df()
                            .add_test(test_ds)
                            .transform(tfms, size=128)
                            .databunch(path='.', bs=128)
                            .normalize(imagenet_stats))
train_ds2.show_batch(rows=4, figsize=(6,6))
train_ds3 = (ImageList.from_df(train_df, path='.', folder='train')
                            .split_by_rand_pct(0.08)
                            .label_from_df()
                            .add_test(test_ds)
                            .transform(tfms, size=128)
                            .databunch(path='.', bs=128)
                            .normalize(imagenet_stats))
train_ds3.show_batch(rows=4, figsize=(6,6))
learn1 = cnn_learner(train_ds1, models.densenet121, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
learn1.freeze()
learn1.lr_find()
learn1.recorder.plot(suggestion=True)
learn1.fit_one_cycle(5, max_lr=slice(1e-3, 1e-2, 1e-1))
learn2 = cnn_learner(train_ds2, models.densenet161, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
learn2.freeze()
learn2.lr_find()
learn2.recorder.plot(suggestion=True, skip_end=0)
learn2.fit_one_cycle(5, max_lr=slice(1e-3, 1e-2, 1e-1))
learn3 = cnn_learner(train_ds3, models.densenet201, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
learn3.freeze()
learn3.lr_find()
learn3.recorder.plot(suggestion=True)
learn3.fit_one_cycle(5, max_lr=slice(1e-3, 1e-2, 1e-1))
learn1.save('before_unfreeze_densenet121')
learn2.save('before_unfreeze_densenet161')
learn3.save('before_unfreeze_densenet201')
learn1.recorder.plot_losses()
learn2.recorder.plot_losses()
learn3.recorder.plot_losses()
preds1,_ = learn1.TTA(ds_type=DatasetType.Test)
preds2,_ = learn2.TTA(ds_type=DatasetType.Test)
preds3,_ = learn3.TTA(ds_type=DatasetType.Test)
np_final = preds1 + preds2 + preds3
np_final = np_final / 3.0
tmp_df['has_cactus'] = np.argmax(np_final, axis=1)
tmp_df.head()
tmp_df.to_csv("submission.csv", index=False)
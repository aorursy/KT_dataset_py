# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pathlib import Path

from fastai import *

from fastai.vision import *

from fastai.callbacks import *

import torch
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_folder = Path("../input")

#data_folder.joinpath('train').ls()
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/sample_submission.csv")
train_df.head(), train_df.info()
test_img = ImageList.from_df(test_df, path=data_folder/'test', folder='test')

trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.2, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)

train_il = ImageList.from_df(train_df, path=data_folder/'train', folder='train')

train_img = (train_il.split_by_rand_pct(0.01)

            .label_from_df()

            .add_test(test_img)

            .transform(trfm, size=336)

            .databunch(path='.', bs=32, device= torch.device('cuda:0'))

            .normalize(imagenet_stats)

           )
train_il
# train_img.show_batch(rows=3, figsize=(7,6))
callbacks = [partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.01, patience=3)]
learn = cnn_learner(train_img, models.densenet161, metrics=[error_rate, accuracy], callback_fns=callbacks).mixup().to_fp16()
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 1e-02

learn.fit_one_cycle(3, slice(lr), callbacks=[SaveModelCallback(learn, every='improvement', monitor='quadratic_kappa', name='bestmodel')])
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(16, slice(1e-05, 1e-03), callbacks=[SaveModelCallback(learn, every='improvement', monitor='quadratic_kappa', name='bestmodel')])
#interp = ClassificationInterpretation.from_learner(learn)

#interp.plot_top_losses(9, figsize=(7,6))
learn.to_fp32()
preds,_ = learn.TTA(ds_type=DatasetType.Test)
test_df.has_cactus = preds.numpy()[:, 0]
test_df.to_csv('submission.csv', index=False)
test_df.head()
pseudo_df = test_df.copy()
pseudo_df.loc[pseudo_df['has_cactus'] > 0.99, 'has_cactus'] = 1

pseudo_df.loc[pseudo_df['has_cactus'] < 0.01, 'has_cactus'] = 0
pseudo_label_df = pseudo_df[pseudo_df['has_cactus'] > 0.99]

pseudo_label_df.append(pseudo_df[pseudo_df['has_cactus'] < 0.01])

pseudo_label_df.shape
pseudo_label_df['has_cactus'] = pseudo_label_df['has_cactus'].astype(np.int64)
pseudo_label_df.head()
label_src = ImageList.from_df(pseudo_label_df, path = data_folder/'test', folder='test', cols='id')
label_src
train_il.add(label_src)
train_img = (train_il.split_by_rand_pct(0.01)

            .label_from_df()

            .add_test(test_img)

            .transform(trfm, size=336)

            .databunch(path='.', bs=32, device= torch.device('cuda:0'))

            .normalize(imagenet_stats)

           )
learn = cnn_learner(train_img, models.densenet161, metrics=[error_rate, accuracy], callback_fns=callbacks).mixup().to_fp16()
lr = 1e-02

learn.fit_one_cycle(3, slice(lr), callbacks=[SaveModelCallback(learn, every='improvement', monitor='quadratic_kappa', name='bestmodel')])
learn.unfreeze()

learn.fit_one_cycle(16, slice(1e-05, 1e-03), callbacks=[SaveModelCallback(learn, every='improvement', monitor='quadratic_kappa', name='bestmodel')])
learn.to_fp32()
preds,_ = learn.TTA(ds_type=DatasetType.Test)

test_df.has_cactus = preds.numpy()[:, 0]

test_df.to_csv('submission.csv', index=False)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# ls ../input/train/train
from fastai import *

from pathlib import Path 

from fastai.vision import *
data_folder = Path("../input/")
train_df = pd.read_csv(f"{data_folder}/train.csv")

test_df = pd.read_csv(f"{data_folder}/sample_submission.csv")
train_df.head()
trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)

# train_img = ImageList.from_df(train_df, path=data_folder, folder='train').split_by_rand_pct(0.01).label_from_df().transform(trfm, size=132).databunch(path='.', bs=64, device= torch.device('cuda:0')).normalize(imagenet_stats)

data = ImageDataBunch.from_df(data_folder/'train/train/', df=train_df[["id", "has_cactus"]], label_col="has_cactus",

                              ds_tfms=get_transforms(), folder="", size=132).normalize(imagenet_stats)       
learn = cnn_learner(data, models.resnet18, metrics=[error_rate, accuracy],model_dir="/tmp/model/")
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2)#, slice(5e-4, 9e-2))
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2)
data.add_test(ImageList.from_df(test_df,data_folder/'test',folder='test'))
preds,_ = learn.get_preds(ds_type=DatasetType.Test)
test_df.has_cactus = preds.numpy()[:, 0]
test_df.to_csv('submission.csv', index=False)
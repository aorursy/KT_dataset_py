!pip install fastai==1.0.52
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
from pathlib import Path

from fastai.vision import *

import torch

import random
PATH = Path('../input')

'''Setting seed for reproducibility'''

SEED = 2019



# python RNG

random.seed(SEED)



# pytorch RNGs

torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)



# numpy RNG

np.random.seed(SEED)
batch_size = 96
tfms = get_transforms(flip_vert=True)
src = (ImageList.from_csv(csv_name='train.csv', path=PATH, folder='train/train')

            .split_by_rand_pct(0.1)

            .label_from_df(cols='has_cactus'))
data = (src.transform(tfms,size=128)

          .databunch(bs=batch_size)

           .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(9, 9))
data.classes, data.c
learn = cnn_learner(data, models.densenet161, metrics=accuracy, path='./')
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
lr = 3e-2

learn.fit_one_cycle(4,slice(lr))
# learn.save('stage-1-dense161')
# learn.unfreeze()
# learn.fit_one_cycle(5, 1e-6)
# learn.save('stage-2-dense161')
# interp = ClassificationInterpretation.from_learner(learn)

# interp.plot_top_losses(9, figsize=(7,6))
# interp.plot_confusion_matrix(figsize=(3, 3))
data = (src.transform(tfms, size=128)

        .add_test_folder('test/test')

        .databunch()

        .normalize(imagenet_stats))
learn.data = data
probs, _ = learn.get_preds(ds_type=DatasetType.Test)
ilst = data.test_ds.x
fnames = [item.name for item in ilst.items]; fnames[:10]
test_df = pd.DataFrame({'id': fnames, 'has_cactus': probs.numpy()[:,0]}); test_df.head()
test_df.to_csv('submission.csv', index=None)
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
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



df_train.head()
path = Path("../input")
np.random.seed(42)

src = (ImageList.from_csv(path, 'train.csv', folder='train_images', suffix='.png')

       .split_by_rand_pct(0.2)

       .label_from_df(label_delim=' '))
data = (src.transform(tfms = get_transforms(), size=128)

        .databunch(bs = 128).normalize(imagenet_stats))
data.show_batch()
def acc(input:Tensor, targs:Tensor)->Rank0Tensor:

    "Computes accuracy with `targs` when `input` is bs * n_classes."

    n = targs.shape[0]

    input = input.argmax(dim=-1).view(n,-1)

    targs = targs.view(n,-1).long()

    return (input==targs).float().mean()
learn = cnn_learner(data, models.resnet34, metrics=[acc], model_dir="/tmp/model/", callback_fns=ShowGraph)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-3))
learn.unfreeze()
# learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, slice(1e-5, 1e-3))
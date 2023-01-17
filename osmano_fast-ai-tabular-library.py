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

import matplotlib.pyplot as plt

import os

from fastai.tabular import *
df = pd.read_csv('../input/train.csv').drop('ID_code', axis=1)

test_df = pd.read_csv('../input/test.csv')

valid_idx = random.sample(list(df.index.values), int(len(df)*0.05))
features = [feature for feature in df.columns if 'var_' in feature]

features
data = TabularDataBunch.from_df(path='.', df=df, dep_var='target', valid_idx=valid_idx,

                                    cat_names=[], cont_names=features, procs=[FillMissing, Normalize], test_df=test_df)
learn = tabular_learner(data, layers=[200, 100, 50], metrics=[accuracy])
learn.lr_find(end_lr=1e3)

learn.recorder.plot()
learn.fit_one_cycle(1, max_lr=5e-2)

learn.recorder.plot_losses()
test_pred, test_y = learn.get_preds(ds_type=DatasetType.Test)
print(test_y[0:20])

test_y.sum()
valid_pred, valid_y = learn.get_preds(ds_type=DatasetType.Valid)

print(valid_y[0:20])

valid_y.sum()
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
from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
torch.cuda.is_available()
torch.backends.cudnn.enabled
PATH = "../input/"
csvf = f"{PATH}train_v2.csv"
n = len(list(open(csvf))) - 1 # header is not counted (-1)
val_idxs = get_cv_idxs(n)
tfms = tfms_from_model(resnet34, 264, aug_tfms=transforms_top_down, max_zoom=1.0)
data = ImageClassifierData.from_csv(PATH, 'train-jpg', f'{PATH}train_v2.csv', test_name='test-jpg-v2', # we need to specify where the test set is if you want to submit to Kaggle competitions
                                   val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=64)
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
learn = ConvLearner.pretrained(resnet34, data, precompute=True,tmp_name = TMP_PATH,models_name=MODEL_PATH)
def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 2, average='samples')
                    for th in np.arange(start,end,step)])
metrics=[f2]
learn.precompute=False
learn.fit(5, 2, cycle_len=1, cycle_mult=2)
multi_preds, y = learn.TTA()
preds = np.mean(multi_preds, 0)

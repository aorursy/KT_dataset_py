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
%matplotlib inline

from fastai.imports import *
from fastai.learner import *
from fastai.column_data import *
torch.cuda.is_available()
PATH = '../input/'
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
csv = pd.read_csv(PATH+'songsDataset.csv', header=0, names=['userID', 'songID', 'rating'])
len(csv)
# validation set
val_idxs = get_cv_idxs(len(csv))
data = CollabFilterDataset.from_data_frame(PATH, csv, 'userID', 'songID', 'rating')
learn = data.get_learner(50, val_idxs, 200, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn
learn.lr_find()
learn.sched.plot()
learn.fit(5e-1, 2, cycle_len=1, cycle_mult=2, wds=1e-4)

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
from matplotlib import pyplot as plt
%matplotlib inline

import seaborn as sns
train = pd.read_csv('../input/training_set.csv')
train_meta = pd.read_csv('../input/training_set_metadata.csv')
train['mjd_frac'] = np.modf(train.mjd)[0]

_ = plt.hist(train.mjd_frac, bins=100)
train['mjd_frac'] = np.modf(train.mjd + 0.3)[0]

_ = plt.hist(train.mjd_frac, bins=100)
train['mjd_night'] = np.modf(train.mjd + 0.3)[1]
train = train.groupby(['object_id', 'mjd_night', 'passband']).mjd.mean().to_frame()
train = train.reset_index()
train.head()
train_ddf = train.merge(train_meta, how='left', on='object_id')
train_ddf = train_ddf[train_ddf.ddf == 1]
passband_0 = set(train_ddf[train_ddf.passband == 0].mjd_night.unique())
passband_1_to_5 = set(train_ddf[train_ddf.passband > 0].mjd_night.unique())
len(passband_0), len(passband_1_to_5)
passband_0 & passband_1_to_5

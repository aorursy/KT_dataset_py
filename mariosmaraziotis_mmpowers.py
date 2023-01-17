# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow_addons as tfa
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from tqdm.notebook import tqdm
train_targets.head()
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')

ss = pd.read_csv('../input/lish-moa/sample_submission.csv')
train_features.head(3)
train_targets.describe()

train_targets.sum(axis=1).hist(bins=100
                              )
metrics = []
for _target in train_targets.columns:
    metrics.append(log_loss(train_targets.loc[:, _target], res.loc[:, _target]))
print(f'OOF Metric: {np.mean(metrics)}')
metrics = []
res.loc[train['cp_type']==1, train_targets.columns] = 0
for _target in train_targets.columns:
    metrics.append(log_loss(train_targets.loc[:, _target], res.loc[:, _target]))
print(f'OOF Metric with postprocessing: {np.mean(metrics)}')
ss.loc[test['cp_type']==1, train_targets.columns] = 0
ss.to_csv('submission.csv', index=False)
train_features.shape
train_targets.shape
test_features.shape
train_features.head()
train_targets.head()
list(train_features.columns)
train_targets.sum(axis=1).max()
ss.head()

ss.shape
list(train_features.columns)
list(train_targets.columns)
ss.head(30)

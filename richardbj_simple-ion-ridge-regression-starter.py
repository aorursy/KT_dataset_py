# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.linear_model import Ridge
train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
train.head()
train.tail()
train.loc[49000:50010,:]
train.shape
train['open_channels'].min()
train_time = train['time'].values
train_time_0 = train_time[:50000]
for i in range(1,100):

    train_time_0 = np.hstack([train_time_0, train_time[i*50000:(i+1)*50000]])
train['time'] = train_time_0
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
test.head()
test.tail()
test.shape
train_time_0 = train_time[:50000]

for i in range(1,40):

    train_time_0 = np.hstack([train_time_0, train_time[i*50000:(i+1)*50000]])

test['time'] = train_time_0
n_groups = 100

train["group"] = 0

for i in range(n_groups):

    ids = np.arange(i*50000, (i+1)*50000)

    train.loc[ids,"group"] = i
n_groups = 40

test["group"] = 0

for i in range(n_groups):

    ids = np.arange(i*50000, (i+1)*50000)

    test.loc[ids,"group"] = i
train['signal_2'] = 0

test['signal_2'] = 0
n_groups = 100

for i in range(n_groups):

    sub = train[train.group == i]

    signals = sub.signal.values

    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))

    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))

    signals = signals*(imax-imin)

    train.loc[sub.index,"signal_2"] = [0,] +list(np.array(signals[:-1]))
n_groups = 40

for i in range(n_groups):

    sub = test[test.group == i]

    signals = sub.signal.values

    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))

    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))

    signals = signals*(imax-imin)

    test.loc[sub.index,"signal_2"] = [0,] +list(np.array(signals[:-1]))
signals.shape
#X = train[['time', 'signal_2']].values

X = train[['signal_2']].values

y = train['open_channels'].values
model = Ridge()

model.fit(X, y)

train_preds = model.predict(X)

train_preds = np.clip(train_preds, 0, 10)

train_preds = train_preds.astype(int)
from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import accuracy_score

gt_labels=train.values[:,2]

print(gt_labels[:5])

print(train_preds[:5])

print(accuracy_score(gt_labels,train_preds))

print(cohen_kappa_score(gt_labels,train_preds))

print(cohen_kappa_score(gt_labels,train_preds,weights="quadratic"))

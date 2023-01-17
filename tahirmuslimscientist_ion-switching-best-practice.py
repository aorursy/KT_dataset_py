import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.linear_model import Ridge
MainDir = '/kaggle/input/liverpool-ion-switching/'
train = pd.read_csv(MainDir + 'train.csv')

test = pd.read_csv(MainDir + 'test.csv')

submission = pd.read_csv(MainDir + 'sample_submission.csv')
train.sample(10)
test.sample(10)
submission.sample(10)
train_time = train['time'].values

train_time_0 = train_time[:50000]

train_time_0 = list(train_time_0)*100

train['time'] = train_time_0
train_time_0 = train_time[:50000]

train_time_0 = list(train_time_0)*40

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
X = train[['time', 'signal_2']].values

y = train['open_channels'].values
model = Ridge()

model.fit(X, y)

train_preds = model.predict(X)
train_preds = np.clip(train_preds, 0, 10)
train_preds = train_preds.astype(int)
X_test = test[['time', 'signal_2']].values
test_preds = model.predict(X_test)

test_preds = np.clip(test_preds, 0, 10)

test_preds = test_preds.astype(int)

submission['open_channels'] = test_preds
np.set_printoptions(precision=4)
submission['time'] = [format(submission.time.values[x], '.4f') for x in range(2000000)]
submission.sample(10)
submission.to_csv('submission.csv', index=False)
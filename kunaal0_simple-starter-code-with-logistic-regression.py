import numpy as np 

import pandas as pd 

import math

from sklearn.linear_model import LogisticRegression
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
train_time = train['time'].values

train_time_0 = train_time[:50000]
for i in range(1,100):

    train_time_0 = np.hstack([train_time_0, train_time[i*50000:(i+1)*50000]])
train['time'] = train_time_0
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
X = train[['signal_2']].values

y = train['open_channels'].values
model = LogisticRegression(max_iter=1000)
model.fit(X,y)
train_preds = model.predict(X)
train_preds = np.clip(train_preds, 0, 10)
train_preds = train_preds.astype(int)
X_test = test[['signal_2']].values
test_preds = model.predict(X_test)

test_preds = np.clip(test_preds, 0, 10)

test_preds = test_preds.astype(int)

submission['open_channels'] = test_preds
np.set_printoptions(precision=4)
submission['time'] = [format(submission.time.values[x], '.4f') for x in range(2000000)]
submission.to_csv('submission.csv', index=False)
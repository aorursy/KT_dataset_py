import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgbm

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

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
train_time_0.shape
train['time'] = train_time_0
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

test.head()
test.tail()
test.shape
train_time_0 = train_time[:50000]

for i in range(1,40):

    train_time_0 = np.hstack([train_time_0, train_time[i*50000:(i+1)*50000]])

test['time'] = train_time_0
X = train[['time', 'signal']].values

y = train['open_channels'].values
model = lgbm.LGBMRegressor(n_estimators=100)

model.fit(X, y)
train_preds = model.predict(X)
train_preds = np.clip(train_preds, 0, 10)

train_preds = train_preds.astype(int)

X_test = test[['time', 'signal']].values
submission.head()
submission.shape
X_test.shape
test_preds = model.predict(X_test)

test_preds = np.clip(test_preds, 0, 10)

test_preds = test_preds.astype(int)

submission['open_channels'] = test_preds
submission.head()
np.set_printoptions(precision=4)
submission.time.values[:20]
submission['time'] = [format(submission.time.values[x], '.4f') for x in range(2000000)]
submission.time.values[:20]
submission['open_channels'].mean()
submission.head()
submission.to_csv('submission.csv', index=False)
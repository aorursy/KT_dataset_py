import numpy as np; import pandas as pd;from scipy import signal

from sklearn.preprocessing import MinMaxScaler,RobustScaler

import warnings;warnings.filterwarnings("ignore");import lightgbm as lgb

train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv');

test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

train_test =pd.concat([train, test],sort=True).reset_index()

train_test_processed=pd.DataFrame()

for i in range(len(train_test)//100000):

    batch =train_test.loc[i * 100000:((i + 1) * 100000) - 1]

    batch['signal'] = signal.detrend(batch.signal)

    batch['roll_min'] = batch['signal'].rolling(window=5, min_periods=1,center=True).min()

    batch['roll_max'] = batch['signal'].rolling(window=5, min_periods=1,center=True).max()

    batch['signal_scaled_mm']=MinMaxScaler().fit_transform(batch['signal'].values.reshape(-1,1)).reshape(-1).mean()

    batch['signal_scaled_rs']=RobustScaler().fit_transform(batch['signal'].values.reshape(-1,1)).reshape(-1).mean()

    train_test_processed=pd.concat([train_test_processed, batch],sort=False).reset_index(drop=True)

y_train = train['open_channels']

X_train=train_test_processed[:len(train)][['signal','signal_scaled_mm','signal_scaled_rs','roll_min','roll_max']]

X_test=train_test_processed[-len(test):][['signal','signal_scaled_mm','signal_scaled_rs','roll_min','roll_max']]

cls = lgb.LGBMClassifier(n_estimators=111, n_jobs= -1); cls.fit(X_train, y_train)

pd.DataFrame({'time': test.time, 'open_channels': cls.predict(X_test)}).to_csv('submission.csv', index=False,float_format='%.4f')
import pandas as pd

import numpy as np

import gc

from sklearn.metrics import mean_absolute_error

from sklearn.svm import SVR
def gen_features(X):

    strain = []

    strain.append(X.values[0])

    strain.append(X.values[-1])

    strain.append(X.mean())

    strain.append(X.count())

    strain.append(X.sum())

    strain.append(X.mad())

    strain.append(X.std())

    strain.append(X.median())

    strain.append(X.min())

    strain.append(X.max())

    strain.append(X.kurtosis())

    strain.append(X.skew())

    strain.append(X.nunique())

    strain.append(X.sem())

    strain.append(np.quantile(X,0.01))

    strain.append(np.quantile(X,0.05))

    strain.append(np.quantile(X,0.95))

    strain.append(np.quantile(X,0.99))

    strain.append(np.abs(X).max())

    strain.append(np.abs(X).mean())

    strain.append(np.abs(X).std())

    return pd.Series(strain)
train = pd.read_csv("../input/train.csv", iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

x_train = pd.DataFrame()

y_train = pd.Series()

for df in train:

    ch = gen_features(df['acoustic_data'])

    x_train = x_train.append(ch, ignore_index=True)

    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))

    

del train

gc.collect()
m = SVR()

m.fit(x_train,y_train) 

mean_absolute_error(y_train,m.predict(x_train))
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

x_test = pd.DataFrame(columns=x_train.columns, dtype=np.float64, index=submission.index)

for seg_id in x_test.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    ch = gen_features(seg['acoustic_data'])

    x_test.loc[seg_id]= ch    

    

submission['time_to_failure'] = m.predict(x_test)

submission.to_csv('submission.csv')
import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15



import lightgbm as lgb

import xgboost as xgb

import time

import datetime



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.linear_model import Ridge, RidgeCV

import gc

from catboost import CatBoostRegressor

import seaborn as sns

import os

print(os.listdir("../input"))

X = pd.read_csv("../input/train.csv", nrows = 600000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
%time



rows = 150_000

train = X

segments = int(np.floor(train.shape[0] / rows))



X_tr = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['ave', 'std', 'max', 'min',

                               'av_change_abs', 'av_change_rate', 'abs_max', 'abs_min',

                               'std_first_50000', 'std_last_50000', 'std_first_10000', 'std_last_10000',

                               'avg_first_50000', 'avg_last_50000', 'avg_first_10000', 'avg_last_10000',

                               'min_first_50000', 'min_last_50000', 'min_first_10000', 'min_last_10000',

                               'max_first_50000', 'max_last_50000', 'max_first_10000', 'max_last_10000'])

y_tr = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])



total_mean = train['acoustic_data'].mean()

total_std = train['acoustic_data'].std()

total_max = train['acoustic_data'].max()

total_min = train['acoustic_data'].min()

total_sum = train['acoustic_data'].sum()

total_abs_max = np.abs(train['acoustic_data']).sum()



for segment in tqdm_notebook(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    x = seg['acoustic_data'].values

    y = seg['time_to_failure'].values[-1]

    

    y_tr.loc[segment, 'time_to_failure'] = y

    X_tr.loc[segment, 'ave'] = x.mean()

    X_tr.loc[segment, 'std'] = x.std()

    X_tr.loc[segment, 'max'] = x.max()

    X_tr.loc[segment, 'min'] = x.min()

    

    

    X_tr.loc[segment, 'av_change_abs'] = np.mean(np.diff(x))

    X_tr.loc[segment, 'av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])

    X_tr.loc[segment, 'abs_max'] = np.abs(x).max()

    X_tr.loc[segment, 'abs_min'] = np.abs(x).min()

    

    X_tr.loc[segment, 'std_first_50000'] = x[:50000].std()

    X_tr.loc[segment, 'std_last_50000'] = x[-50000:].std()

    X_tr.loc[segment, 'std_first_10000'] = x[:10000].std()

    X_tr.loc[segment, 'std_last_10000'] = x[-10000:].std()

    

    X_tr.loc[segment, 'avg_first_50000'] = x[:50000].mean()

    X_tr.loc[segment, 'avg_last_50000'] = x[-50000:].mean()

    X_tr.loc[segment, 'avg_first_10000'] = x[:10000].mean()

    X_tr.loc[segment, 'avg_last_10000'] = x[-10000:].mean()

    

    X_tr.loc[segment, 'min_first_50000'] = x[:50000].min()

    X_tr.loc[segment, 'min_last_50000'] = x[-50000:].min()

    X_tr.loc[segment, 'min_first_10000'] = x[:10000].min()

    X_tr.loc[segment, 'min_last_10000'] = x[-10000:].min()

    

    X_tr.loc[segment, 'max_first_50000'] = x[:50000].max()

    X_tr.loc[segment, 'max_last_50000'] = x[-50000:].max()

    X_tr.loc[segment, 'max_first_10000'] = x[:10000].max()

    X_tr.loc[segment, 'max_last_10000'] = x[-10000:].max()
scaler = StandardScaler()

scaler.fit(X_tr)

X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_test = pd.DataFrame(columns=X_tr.columns, dtype=np.float64, index=submission.index)

plt.figure(figsize=(22, 16))



for i, seg_id in enumerate(tqdm_notebook(X_test.index)):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = seg['acoustic_data'].values

    X_test.loc[seg_id, 'ave'] = x.mean()

    X_test.loc[seg_id, 'std'] = x.std()

    X_test.loc[seg_id, 'max'] = x.max()

    X_test.loc[seg_id, 'min'] = x.min()

        

    X_test.loc[seg_id, 'av_change_abs'] = np.mean(np.diff(x))

    X_test.loc[seg_id, 'av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])

    X_test.loc[seg_id, 'abs_max'] = np.abs(x).max()

    X_test.loc[seg_id, 'abs_min'] = np.abs(x).min()

    

    X_test.loc[seg_id, 'std_first_50000'] = x[:50000].std()

    X_test.loc[seg_id, 'std_last_50000'] = x[-50000:].std()

    X_test.loc[seg_id, 'std_first_10000'] = x[:10000].std()

    X_test.loc[seg_id, 'std_last_10000'] = x[-10000:].std()

    

    X_test.loc[seg_id, 'avg_first_50000'] = x[:50000].mean()

    X_test.loc[seg_id, 'avg_last_50000'] = x[-50000:].mean()

    X_test.loc[seg_id, 'avg_first_10000'] = x[:10000].mean()

    X_test.loc[seg_id, 'avg_last_10000'] = x[-10000:].mean()

    

    X_test.loc[seg_id, 'min_first_50000'] = x[:50000].min()

    X_test.loc[seg_id, 'min_last_50000'] = x[-50000:].min()

    X_test.loc[seg_id, 'min_first_10000'] = x[:10000].min()

    X_test.loc[seg_id, 'min_last_10000'] = x[-10000:].min()

    

    X_test.loc[seg_id, 'max_first_50000'] = x[:50000].max()

    X_test.loc[seg_id, 'max_last_50000'] = x[-50000:].max()

    X_test.loc[seg_id, 'max_first_10000'] = x[:10000].max()

    X_test.loc[seg_id, 'max_last_10000'] = x[-10000:].max()

    

X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
model = lgb.LGBMRegressor(n_estimators=10000,learning_rate=0.0005, n_jobs=-1)

model.fit(X_train_scaled,y_tr)

y_pred_lgb = model.predict(X_test_scaled)
sample = pd.read_csv("../input/sample_submission.csv")

sample['time_to_failure'] = y_pred_lgb

sample.to_csv('submission.csv',index=False)
%time



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss, roc_auc_score



n_estimators = [5000,10000,15000]

learning_rates = [0.0001,0.0005, 0.001]

for learning_rate in learning_rates:

  for n_est in  n_estimators:

    print(n_est)

    print(learning_rate)

    mae =[]

    cv = KFold(n_splits=5, shuffle=True,random_state=0)

    for train, valid in cv.split(X_train_scaled, y_tr):

        x_train = X_train_scaled.iloc[train]

        x_valid = X_train_scaled.iloc[valid]

        y_train = y_tr.iloc[train]

        y_valid = y_tr.iloc[valid]

        model = lgb.LGBMRegressor(n_estimators=n_est,learning_rate=learning_rate, n_jobs=-1,random_state=0)

        model.fit(x_train, y_train)

        y_pred_lgb = model.predict(x_valid)

        y_pred = y_pred_lgb

        mae.append(mean_absolute_error(y_valid, y_pred))

    print(mean(mae))
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



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

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



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
model = lgb.LGBMRegressor(n_estimators=1000, n_jobs=-1)

model.fit(X_train_scaled,y_tr)

y_pred_lgb = model.predict(X_test_scaled)
import xgboost as xgb



model = xgb.XGBRegressor(n_estimators=1000)

model.fit(X_train_scaled,y_tr)

y_pred_xgb = model.predict(X_test_scaled)
sample = pd.read_csv("../input/sample_submission.csv")

sample['time_to_failure'] = (y_pred_lgb+y_pred_xgb)/2

sample.to_csv('submission.csv',index=False)
%time



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss, roc_auc_score



n_estimators = [10,100,1000]

for n_est in  n_estimators:

    print(n_est)

    cv = KFold(n_splits=5, shuffle=True,random_state=0)

    for train, valid in cv.split(X_train_scaled, y_tr):

        x_train = X_train_scaled.iloc[train]

        x_valid = X_train_scaled.iloc[valid]

        y_train = y_tr.iloc[train]

        y_valid = y_tr.iloc[valid]

        model_1 = lgb.LGBMRegressor(n_estimators=n_est, n_jobs=-1,random_state=0)

        model_1.fit(x_train, y_train)

        y_pred_lgb = model_1.predict(x_valid)

        model_2 = xgb.XGBRegressor(n_estimators=n_est, n_jobs=-1,random_state=0)

        model_2.fit(x_train, y_train)

        y_pred_xgb = model_2.predict(x_valid)

        y_pred = (y_pred_lgb + y_pred_xgb)/2

        print(mean_absolute_error(y_valid, y_pred))  
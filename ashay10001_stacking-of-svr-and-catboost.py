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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.metrics import mean_absolute_error, make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import BayesianRidge
train=pd.read_csv("../input/train.csv",dtype={"acoustic_data": np.int16, "time_to_failure": np.float64})

rows = 150000

segments = int(np.floor(train.shape[0] / rows))
col_names = ['mean','max','variance','min', 'stdev', 'q1', 'q5', 'q95', 'q99']
X1= pd.DataFrame(index=range(segments), dtype=np.float64, columns=col_names)

Y1 = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
for segment in tqdm(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    x = seg['acoustic_data'].values

    y = seg['time_to_failure'].values[-1]

    Y1.loc[segment, 'time_to_failure'] = y

    X1.loc[segment, 'mean'] = x.mean()

    X1.loc[segment, 'stdev'] = x.std()

    X1.loc[segment, 'variance'] = np.var(x)

    X1.loc[segment, 'max'] = x.max()

    X1.loc[segment, 'min'] = x.min()

    X1.loc[segment, 'q1'] = np.quantile(x, 0.01)

    X1.loc[segment, 'q5'] = np.quantile(x, 0.05)

    X1.loc[segment, 'q95'] = np.quantile(x, 0.95)

    X1.loc[segment, 'q99'] = np.quantile(x, 0.99)  

    z = np.fft.fft(x)

    realFFT = np.real(z)

    imagFFT = np.imag(z)

    X1.loc[segment, 'A0'] = abs(z[0])

    X1.loc[segment, 'Real_mean'] = realFFT.mean()

    X1.loc[segment, 'Real_std'] = realFFT.std()

    X1.loc[segment, 'Real_max'] = realFFT.max()

    X1.loc[segment, 'Real_min'] = realFFT.min()

    X1.loc[segment, 'Imag_mean'] = imagFFT.mean()

    X1.loc[segment, 'Imag_std'] = imagFFT.std()

    X1.loc[segment, 'Imag_max'] = imagFFT.max()

    X1.loc[segment, 'Imag_min'] = imagFFT.min()

    

X1.describe()
sub=pd.read_csv("../input/sample_submission.csv",index_col='seg_id')

xtest=pd.DataFrame(columns=X1.columns,dtype=np.float64,index=sub.index)

xtest.describe()
for i, seg_id in enumerate(tqdm(xtest.index)):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = pd.Series(seg['acoustic_data'].values)

    z = np.fft.fft(x)

    realFFT = np.real(z)

    imagFFT = np.imag(z)

    

    xtest.loc[seg_id, 'mean'] = x.mean()

    xtest.loc[seg_id, 'stdev'] = x.std()

    xtest.loc[seg_id, 'variance'] = np.var(x)

    xtest.loc[seg_id, 'max'] = x.max()

    xtest.loc[seg_id, 'min'] = x.min()

    xtest.loc[seg_id, 'q1'] = np.quantile(x, 0.01)

    xtest.loc[seg_id, 'q5'] = np.quantile(x, 0.05)

    xtest.loc[seg_id, 'q95'] = np.quantile(x, 0.95)

    xtest.loc[seg_id, 'q99'] = np.quantile(x, 0.99)

    xtest.loc[seg_id, 'A0'] = abs(z[0])

    xtest.loc[seg_id, 'Real_mean'] = realFFT.mean()

    xtest.loc[seg_id, 'Real_std'] = realFFT.std()

    xtest.loc[seg_id, 'Real_max'] = realFFT.max()

    xtest.loc[seg_id, 'Real_min'] = realFFT.min()

    xtest.loc[seg_id, 'Imag_mean'] = imagFFT.mean()

    xtest.loc[seg_id, 'Imag_std'] = imagFFT.std()

    xtest.loc[seg_id, 'Imag_max'] = imagFFT.max()

    xtest.loc[seg_id, 'Imag_min'] = imagFFT.min()
sc=StandardScaler()

sc.fit(X1)

scX = pd.DataFrame(sc.transform(X1), columns = X1.columns)

sctestx = pd.DataFrame(sc.transform(xtest), columns = xtest.columns)

sctestx.shape
parameters = {'num_leaves': 31,'min_data_in_leaf': 32, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.001,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "nthread": 4,

         "verbosity": -1}
import lightgbm as lgb

features=scX.columns

folds = KFold(n_splits=5, random_state = 10,shuffle = True)



oof_clf1 = np.zeros(len(scX))

pred1=np.zeros(len(sctestx))

feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(scX.values, Y1.values)):

    print("fold n°{}".format(fold_))

    trn_data = lgb.Dataset(scX.iloc[trn_idx][features], label=Y1.iloc[trn_idx])

    val_data = lgb.Dataset(scX.iloc[val_idx][features], label=Y1.iloc[val_idx])

    num_round = 10000

    model1=lgb.train(parameters, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)

    oof_clf1[val_idx] = model1.predict(scX.iloc[val_idx][features], num_iteration=model1.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = features

    fold_importance_df["importance"] = model1.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    pred1 += model1.predict(sctestx[features], num_iteration=model1.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof_clf1, Y1)**0.5))

import xgboost as xgb



xgb_params = {'eta': 0.001, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'alpha':0.1,

          'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': True, 'random_state':folds}





folds = KFold(n_splits=5, random_state=4520)

oof_xgb = np.zeros(len(scX))

pred2 = np.zeros(len(sctestx))



for fold_, (trn_idx, val_idx) in enumerate(folds.split(scX.values, Y1.values)):

    print("fold n°{}".format(fold_ + 1))

    trn_data = xgb.DMatrix(data=scX.iloc[trn_idx][features], label=Y1.iloc[trn_idx])

    val_data = xgb.DMatrix(data=scX.iloc[val_idx][features], label=Y1.iloc[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid')]

    print("-" * 10 + "Xgboost " + str(fold_) + "-" * 10)

    num_round = 11000

    xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=50, verbose_eval=1000)

    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(scX.iloc[val_idx][features]), ntree_limit=xgb_model.best_ntree_limit+50)



    pred2 += xgb_model.predict(xgb.DMatrix(sctestx[features]), ntree_limit=xgb_model.best_ntree_limit+50) / folds.n_splits

    

np.save('oof_xgb', oof_xgb)

np.save('predictions_xgb', pred2)

print("CV score: {:<8.5f}".format(mean_squared_error(oof_xgb, Y1)**0.5))
train_stack = np.vstack([oof_clf1, oof_xgb]).transpose()

test_stack = np.vstack([pred1,pred2]).transpose()



folds = KFold(n_splits=5, shuffle=True, random_state=15)

oof_stack = np.zeros(train_stack.shape[0])

prediction = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(scX, Y1)):

    print("fold n°{}".format(fold_))

    trn_data, trn_y = train_stack[trn_idx], Y1.iloc[trn_idx].values

    val_data, val_y = train_stack[val_idx], Y1.iloc[val_idx].values



    print("-" * 10 + "Ridge Regression" + str(fold_) + "-" * 10)

    smodel = BayesianRidge()

    smodel.fit(trn_data, trn_y)

    

    oof_stack[val_idx] = smodel.predict(val_data)

    prediction += smodel.predict(test_stack) / 5

print("CV score: {:<8.5f}".format(mean_squared_error(oof_stack, Y1)**0.5))

sub['time_to_failure'] = prediction

sub.to_csv('fianlprediction.csv')

print(sub.head())
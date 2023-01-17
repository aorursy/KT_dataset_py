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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.isnull().sum().any()
target = train['target']

f = train.drop(['target', 'ID_code'] , axis = 1)

testf = test.drop(['ID_code'], axis = 1 )

from sklearn.decomposition import PCA, FastICA

pca = PCA(n_components=12, random_state=420)

pca2_results_train = pca.fit_transform(f)

pca2_results_test = pca.transform(testf)
ica = FastICA(n_components=12, random_state=420)

ica2_results_train = ica.fit_transform(f)

ica2_results_test = ica.transform(testf)
trainNN = pd.DataFrame()

testNN = pd.DataFrame()
for i in range(1, 13):

    train['pca_' + str(i)] = pca2_results_train[:,i-1]

    test['pca_' + str(i)] = pca2_results_test[:, i-1]



    train['ica_' + str(i)] = ica2_results_train[:,i-1]

    test['ica_' + str(i)] = ica2_results_test[:, i-1]
from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(n_components=15, random_state=420)

tsvd_results_train = tsvd.fit_transform(f)

tsvd_results_test = tsvd.transform(testf)
from sklearn.random_projection import GaussianRandomProjection

grp = GaussianRandomProjection(n_components=15, eps=0.2, random_state=420)

grp_results_train = grp.fit_transform(f)

grp_results_test = grp.transform(testf)
from sklearn.random_projection import SparseRandomProjection

srp = SparseRandomProjection(n_components=15, dense_output=True, random_state=420)

srp_results_train = srp.fit_transform(f)

srp_results_test = srp.transform(testf)
for i in range(1, 16):

    trainNN['tsvd_' + str(i)] = tsvd_results_train[:,i-1]

    testNN['tsvd_' + str(i)] = tsvd_results_test[:, i-1]



    trainNN['grp_' + str(i)] = grp_results_train[:,i-1]

    testNN['grp_' + str(i)] = grp_results_test[:, i-1]



    trainNN['srp_' + str(i)] = srp_results_train[:,i-1]

    testNN['srp_' + str(i)] = srp_results_test[:, i-1]

import xgboost as xgb



# prepare dict of params for xgboost to run with

xgb_params = {

    'n_trees': 520,

    'eta': 0.0045,

    'max_depth': 4,

    'subsample': 0.98,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}
dtrain = xgb.DMatrix(f, target)

dtest = xgb.DMatrix(testf)

num_boost_rounds = 300
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
y_pred = model.predict(dtest)
trainNN.shape
import xgboost as xgb



# prepare dict of params for xgboost to run with

xgb_paramst = {

    'n_trees': 300,

    'eta': 0.0045,

    'max_depth': 4,

    'subsample': 0.5,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}
dttrain = xgb.DMatrix(trainNN, target)

dttest = xgb.DMatrix(testNN)
num_boost_roundst = 800
model2 = xgb.train(dict(xgb_paramst, silent=0), dttrain, num_boost_round=num_boost_rounds)
t_pred = model2.predict(dttest)
sub = pd.DataFrame()

sub['ID_code'] = test['ID_code']

sub['target'] = y_pred*0.65 + t_pred*0.35

sub.to_csv('stacked-models.csv', index=False)
sub.head()
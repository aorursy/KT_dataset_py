# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import pandas as pd

import numpy as np

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.decomposition import PCA

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import r2_score

import xgboost as xgb



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



y_train = train['y'].values

id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])

df_all.drop(['y'], axis=1, inplace=True)



# One-hot encoding of categorical/strings

df_all = pd.get_dummies(df_all, drop_first=True)

df_all.shape
from sklearn.decomposition import PCA, FastICA

n_comp = 10



# PCA

pca = PCA(n_components=n_comp, random_state=42)

pca2_results_train = pca.fit_transform(df_all)

#pca2_results_test = pca.transform(test)



# ICA

ica = FastICA(n_components=n_comp, random_state=42)

ica2_results_train = ica.fit_transform(df_all)

#ica2_results_test = ica.transform(test)



# Append decomposition components to datasets

for i in range(1, n_comp+1):

    df_all['pca_' + str(i)] = pca2_results_train[:,i-1]

    #test['pca_' + str(i)] = pca2_results_test[:, i-1]

    

    df_all['ica_' + str(i)] = ica2_results_train[:,i-1]

    #test['ica_' + str(i)] = ica2_results_test[:, i-1]

    

#y_train = train["y"]

y_mean = np.mean(y_train)
train = df_all[:num_train]

test = df_all[num_train:]
import xgboost as xgb



# prepare dict of params for xgboost to run with

xgb_params = {

    'n_trees': 500, 

    'eta': 0.005,

    'max_depth': 4,

    'subsample': 0.95,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': y_mean, # base prediction = mean(target)

    'silent': 1

}



# form DMatrices for Xgboost training

dtrain = xgb.DMatrix(train, y_train)

dtest = xgb.DMatrix(test)



# xgboost, cross-validation

cv_result = xgb.cv(xgb_params, 

                   dtrain, 

                   num_boost_round=2000, # increase to have better results (~700)

                   early_stopping_rounds=50,

                   verbose_eval=50, 

                   show_stdv=False

                  )



num_boost_rounds = len(cv_result)

print(num_boost_rounds)



# train model

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
# check f2-score (to get higher score - increase num_boost_round in previous cell)

from sklearn.metrics import r2_score



# now fixed, correct calculation

print(r2_score(dtrain.get_label(), model.predict(dtrain)))
# make predictions and save results

y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})

output.to_csv('xgboost-dummy.csv'.format(xgb_params['max_depth']), index=False)
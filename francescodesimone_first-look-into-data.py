# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.decomposition import TruncatedSVD

import xgboost as xgb
plt.figure(figsize=(12,8))

sns.distplot(train_df.y.values, bins=50, kde=False)

plt.xlabel('AVG Time on Test platform', fontsize=12)

plt.show()
# read datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# process columns, apply LabelEncoder to categorical features

for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder() 

        lbl.fit(list(train[c].values) + list(test[c].values)) 

        train[c] = lbl.transform(list(train[c].values))

        test[c] = lbl.transform(list(test[c].values))



# shape        

print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))
##Add decomposed components: PCA / ICA etc.



n_comp = 10



# tSVD

tsvd = TruncatedSVD(n_components=n_comp, random_state=420)

tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))

tsvd_results_test = tsvd.transform(test)



# PCA

pca = PCA(n_components=n_comp, random_state=420)

pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))

pca2_results_test = pca.transform(test)



# ICA

ica = FastICA(n_components=n_comp, random_state=420)

ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))

ica2_results_test = ica.transform(test)



# GRP

grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)

grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))

grp_results_test = grp.transform(test)



# SRP

srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)

srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))

srp_results_test = srp.transform(test)



# Append decomposition components to datasets

for i in range(1, n_comp+1):

    train['pca_' + str(i)] = pca2_results_train[:,i-1]

    test['pca_' + str(i)] = pca2_results_test[:, i-1]

    

    train['ica_' + str(i)] = ica2_results_train[:,i-1]

    test['ica_' + str(i)] = ica2_results_test[:, i-1]



    train['tsvd_' + str(i)] = tsvd_results_train[:,i-1]

    test['tsvd_' + str(i)] = tsvd_results_test[:, i-1]

    

    train['grp_' + str(i)] = grp_results_train[:,i-1]

    test['grp_' + str(i)] = grp_results_test[:, i-1]

    

    train['srp_' + str(i)] = srp_results_train[:,i-1]

    test['srp_' + str(i)] = srp_results_test[:, i-1]

    

y_train = train["y"]

y_mean = np.mean(y_train)





# prepare dict of params for xgboost to run with

xgb_params = {

    'n_trees': 700, 

    'eta': 0.005,

    'max_depth': 4,

    'subsample': 0.8,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': y_mean, 

    'silent': 1

}



# form DMatrices for Xgboost training

dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)

dtest = xgb.DMatrix(test)



# xgboost, cross-validation

cv_result = xgb.cv(xgb_params, 

                   dtrain, 

                   num_boost_round=1500, # increase to have better results (~700)

                   early_stopping_rounds=150,

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
15# make predictions and save results

y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})

output.to_csv('xgboost_last.csv'.format(xgb_params['max_depth']), index=False)


plt.figure(figsize=(12,8))

sns.distplot(y_pred, bins=50, kde=False)

plt.xlabel('Predicted AVG Time on Test platform', fontsize=12)

plt.show()
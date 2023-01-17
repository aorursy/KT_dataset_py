import numpy as np

import pandas as pd

import xgboost as xgb

import gc

from sklearn.linear_model import ElasticNetCV, LassoLarsCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import make_pipeline, make_union

from sklearn.utils import check_array

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.decomposition import PCA, FastICA

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import r2_score

train = pd.read_csv('../input/train_2016.csv')

prop = pd.read_csv('../input/properties_2016.csv')

sample = pd.read_csv('../input/sample_submission.csv')



for c, dtype in zip(prop.columns, prop.dtypes):

	if dtype == np.float64:

		prop[c] = prop[c].astype(np.float32)







df_train = train.merge(prop, how='left', on='parcelid')



x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode', 'basementsqft', 'buildingclasstypeid', 'finishedsquarefeet13', 'storytypeid'], axis=1)

y_train = df_train['logerror'].values









train_columns = x_train.columns



for c in x_train.dtypes[x_train.dtypes == object].index.values:

    x_train[c] = (x_train[c] == True)

    

    

print(x_train.shape, y_train.shape)



#clean data 

x_train = x_train.fillna(0.0)

x_train.head()
n_comp = 12



# tSVD

tsvd = TruncatedSVD(n_components=n_comp, random_state=420)

tsvd_results_train = tsvd.fit_transform(x_train)





# PCA

pca = PCA(n_components=n_comp, random_state=420)

pca2_results_train = pca.fit_transform(x_train)





# ICA

ica = FastICA(n_components=n_comp, random_state=420)

ica2_results_train = ica.fit_transform(x_train)





# GRP

grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)

grp_results_train = grp.fit_transform(x_train)



# SRP

srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)

srp_results_train = srp.fit_transform(x_train)



# Append decomposition components to datasets

for i in range(1, n_comp + 1):

    x_train['pca_' + str(i)] = pca2_results_train[:, i - 1]

    #test['pca_' + str(i)] = pca2_results_test[:, i - 1]



    x_train['ica_' + str(i)] = ica2_results_train[:, i - 1]

    #test['ica_' + str(i)] = ica2_results_test[:, i - 1]



    x_train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]

    #test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]



    x_train['grp_' + str(i)] = grp_results_train[:, i - 1]

    #test['grp_' + str(i)] = grp_results_test[:, i - 1]



    x_train['srp_' + str(i)] = srp_results_train[:, i - 1]

    #test['srp_' + str(i)] = srp_results_test[:, i - 1]

x_train.head()
split = 80000

x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]



print('building data matrix...')



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)





print('training ...')

params = {}

params['eta'] = 0.02

params['objective'] = 'reg:linear'

params['eval_metric'] = ['rmse', 'logloss']

params['max_depth'] = 9

params['silent'] = 1

params['subsample'] = 0.7

params['colsample_bytree'] = 0.7



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



cv_output = xgb.cv(params, d_train, num_boost_round=3000, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=False)



num_boost_rounds = len(cv_output)



clf = xgb.train(params, d_train, num_boost_round= num_boost_rounds)



del d_train, d_valid



print('Building test set ...')



sample['parcelid'] = sample['ParcelId']

df_test = sample.merge(prop, on='parcelid', how='left')



del prop; gc.collect()



x_test = df_test[train_columns]

for c in x_test.dtypes[x_test.dtypes == object].index.values:

    x_test[c] = (x_test[c] == True)



del df_test, sample; gc.collect()



d_test = xgb.DMatrix(x_test)



del x_test; gc.collect()



print('Predicting on test ...')



p_test = clf.predict(d_test)



del d_test; gc.collect()



result = pd.read_csv('../input/sample_submission.csv')



for c in result.columns[result.columns != 'ParcelId']:

    result[c] = p_test



print('writing csv output ...')

result.to_csv('predictions.csv', index=False, float_format='%.4f') # Thanks to @inversion
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection



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

from sklearn.decomposition import PCA, FastICA

from sklearn.decomposition import TruncatedSVD

n_comp = 12



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



from sklearn.cluster import KMeans

km=KMeans(n_clusters=n_comp, random_state=420)

km_train = km.fit_transform(train.drop(["y"],axis=1))

km_test=km.transform(test)

print(km_train.shape)
mean_x0 = train[['X0', 'y']].groupby(['X0'], as_index=False).median()

mean_x0.columns = ['X0', 'mean_x0']



train = pd.merge(train, mean_x0, on='X0', how='outer')



mean_x1 = train[['X1', 'y']].groupby(['X1'], as_index=False).median()

mean_x1.columns = ['X1', 'mean_x1']



train = pd.merge(train, mean_x1, on='X1', how='outer')



mean_x2 = train[['X2', 'y']].groupby(['X2'], as_index=False).median()

mean_x2.columns = ['X2', 'mean_x2']



train = pd.merge(train, mean_x2, on='X2', how='outer')



mean_x3 = train[['X3', 'y']].groupby(['X3'], as_index=False).median()

mean_x3.columns = ['X3', 'mean_x3']



train = pd.merge(train, mean_x3, on='X3', how='outer')



mean_x4 = train[['X4', 'y']].groupby(['X4'], as_index=False).median()

mean_x4.columns = ['X4', 'mean_x4']



train = pd.merge(train, mean_x4, on='X4', how='outer')



mean_x5 = train[['X5', 'y']].groupby(['X5'], as_index=False).median()

mean_x5.columns = ['X5', 'mean_x5']



train = pd.merge(train, mean_x5, on='X5', how='outer')



mean_x6 = train[['X6', 'y']].groupby(['X6'], as_index=False).median()

mean_x6.columns = ['X6', 'mean_x6']



train = pd.merge(train, mean_x6, on='X6', how='outer')



mean_x8 = train[['X8', 'y']].groupby(['X8'], as_index=False).median()

mean_x8.columns = ['X8', 'mean_x8']



train = pd.merge(train, mean_x8, on='X8', how='outer')



train.head(100)

test = pd.merge(test, mean_x0, on='X0', how='left')



test['mean_x0'].fillna(test['mean_x0'].dropna().median(), inplace=True)



test = pd.merge(test, mean_x1, on='X1', how='left')



test['mean_x1'].fillna(test['mean_x1'].dropna().median(), inplace=True)



test = pd.merge(test, mean_x2, on='X2', how='left')



test['mean_x2'].fillna(test['mean_x2'].dropna().median(), inplace=True)



test = pd.merge(test, mean_x3, on='X3', how='left')



test['mean_x3'].fillna(test['mean_x3'].dropna().median(), inplace=True)



test = pd.merge(test, mean_x4, on='X4', how='left')



test['mean_x4'].fillna(test['mean_x4'].dropna().median(), inplace=True)



test = pd.merge(test, mean_x5, on='X5', how='left')



test['mean_x5'].fillna(test['mean_x5'].dropna().median(), inplace=True)



test = pd.merge(test, mean_x6, on='X6', how='left')



test['mean_x6'].fillna(test['mean_x6'].dropna().median(), inplace=True)



test = pd.merge(test, mean_x8, on='X8', how='left')



test['mean_x8'].fillna(test['mean_x8'].dropna().median(), inplace=True)



test.head(1)
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

    

    train['km_'+str(i)]=km_train[:,i-1]

    test['km_'+str(i)]=km_test[:,i-1]







y_train = train["y"]

y_mean = np.mean(y_train)







### Regressor

import xgboost as xgb



# prepare dict of params for xgboost to run with

xgb_params = {

    'n_trees': 520, 

    'eta': 0.0045,

    'max_depth': 4,

    'subsample': 0.93,

    'objective': 'reg:linear',

    'base_score': y_mean, # base prediction = mean(target)

    'silent': 1

}



# form DMatrices for Xgboost training

dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)

dtest = xgb.DMatrix(test)





num_boost_rounds = 1250

# train model

from sklearn.metrics import r2_score

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds,feval=r2_score,maximize=True)





# check f2-score (to get higher score - increase num_boost_round in previous cell)

from sklearn.metrics import r2_score

print(r2_score(dtrain.get_label(),model.predict(dtrain)))



# make predictions and save results

y_pred = model.predict(dtest)



output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})

output.to_csv('submission.csv', index=False)
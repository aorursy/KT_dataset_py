import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

import matplotlib as plt
# read datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# process columns, apply LabelEncoder to categorical features

#for c in train.columns:

#    if train[c].dtype == 'object':

#        lbl = LabelEncoder() 

#        lbl.fit(list(train[c].values) + list(test[c].values)) 

#        train[c] = lbl.transform(list(train[c].values))

#        test[c] = lbl.transform(list(test[c].values))



# shape        

print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

#train = train[train.y < 120]

train.shape
from sklearn.decomposition import PCA, FastICA

n_comp = 20

print(train)

print(np.corrcoef(train))

# PCA

pca = PCA(n_components=n_comp, random_state=42)

pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))

#print(pca.explained_variance_ratio_)

pca2_results_test = pca.transform(test)

print(pca2_results_train.shape)

# ICA

ica = FastICA(n_components=n_comp, random_state=42)

ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))

ica2_results_test = ica.transform(test)



# Append decomposition components to datasets

for i in range(1, n_comp+1):

    train['pca_' + str(i)] = pca2_results_train[:,i-1]

    test['pca_' + str(i)] = pca2_results_test[:, i-1]

    

    train['ica_' + str(i)] = ica2_results_train[:,i-1]

    test['ica_' + str(i)] = ica2_results_test[:, i-1]

    

y_train = train["y"]

y_mean = np.mean(y_train)
from sklearn.ensemble import RandomForestRegressor

regr_rf = RandomForestRegressor(max_depth=30, random_state=2)

regr_rf.fit(train.drop('y',axis=1),y_train)
# check f2-score (to get higher score - increase num_boost_round in previous cell)

from sklearn.metrics import r2_score



# now fixed, correct calculation

print(r2_score(dtrain.get_label(), model.predict(dtrain)))
# make predictions and save results

print("Original test shape: " + str(test.shape))

print("Dtest: " + str(dtest.num_row()) + " , " + str(dtest.num_col()))

y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})

output.to_csv('xgboost-depth{}-pca-ica.csv'.format(xgb_params['max_depth']), index=False)
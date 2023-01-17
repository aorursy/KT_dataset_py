# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, minmax_scale



import matplotlib.pyplot as plt



from sklearn.metrics import mean_squared_error, r2_score



from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score

from sklearn.linear_model import LassoLarsCV, SGDRegressor



from sklearn.svm import SVR, LinearSVC



from sklearn.decomposition import PCA, FastICA

from sklearn.decomposition import TruncatedSVD

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection



from sklearn.neural_network import MLPRegressor

from sklearn.neighbors import KNeighborsRegressor



import xgboost as xgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



train_y = train_df['y']

train_id = train_df['ID']

train_df = train_df.drop("y", 1)

train_df = train_df.drop("ID", 1)



test_id = test_df['ID']

test_df = test_df.drop("ID", 1)



num_train = len(train_df)



df_all = pd.concat([train_df, test_df])

df_all = pd.get_dummies(df_all, drop_first=True)



train_df = df_all[:num_train]

test_df = df_all[num_train:]



#############################



n_comp = 12



# tSVD

tsvd = TruncatedSVD(n_components=n_comp, random_state=420)

tsvd_results_train = tsvd.fit_transform(train_df)

tsvd_results_test = tsvd.transform(test_df)



# PCA

pca = PCA(n_components=n_comp, random_state=420)

pca2_results_train = pca.fit_transform(train_df)

pca2_results_test = pca.transform(test_df)



# ICA

ica = FastICA(n_components=n_comp, random_state=420)

ica2_results_train = ica.fit_transform(train_df)

ica2_results_test = ica.transform(test_df)



# GRP

grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)

grp_results_train = grp.fit_transform(train_df)

grp_results_test = grp.transform(test_df)



# SRP

srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)

srp_results_train = srp.fit_transform(train_df)

srp_results_test = srp.transform(test_df)



# Append decomposition components to datasets

for i in range(1, n_comp+1):

    train_df['pca_' + str(i)] = pca2_results_train[:,i-1]

    test_df['pca_' + str(i)] = pca2_results_test[:, i-1]

    

    train_df['ica_' + str(i)] = ica2_results_train[:,i-1]

    test_df['ica_' + str(i)] = ica2_results_test[:, i-1]



    train_df['tsvd_' + str(i)] = tsvd_results_train[:,i-1]

    test_df['tsvd_' + str(i)] = tsvd_results_test[:, i-1]

    

    train_df['grp_' + str(i)] = grp_results_train[:,i-1]

    test_df['grp_' + str(i)] = grp_results_test[:, i-1]

    

    train_df['srp_' + str(i)] = srp_results_train[:,i-1]

    test_df['srp_' + str(i)] = srp_results_test[:, i-1]



X_dtrain, X_test, y_dtrain, y_test = train_test_split(train_df, train_y, random_state=7, test_size=0.3)
model_rfr = RandomForestRegressor(n_estimators=600, max_depth=3, min_samples_split=4, min_samples_leaf=60)



# Let's see the feature importance for this model

importances = model_rfr.fit(train_df, train_y).feature_importances_

features = pd.DataFrame()

features['feature'] = train_df.columns

features['importance'] = importances



todrop = features.loc[features['importance'] == 0].feature

new_train_df = train_df.drop(todrop, 1)
model_ls = KNeighborsRegressor(n_neighbors=15, weights='distance', algorithm='auto', p=1)





param_test1 = {

 "p": (1, 2)

}



gsearch1 = GridSearchCV(estimator = model_ls, 

 param_grid = param_test1, scoring='r2',iid=False, cv=5)


gsearch1.fit(new_train_df,train_y)

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
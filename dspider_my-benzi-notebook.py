# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, minmax_scale

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LassoLarsCV, ElasticNet, SGDRegressor

from sklearn.tree import ExtraTreeRegressor

from sklearn.svm import SVR

from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

from sklearn.neural_network import MLPRegressor

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

features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

#features[features.size-100:].plot(kind='barh', figsize=(12,24))


#results = cross_val_score(model_rfr, train_df, train_y, cv=10)
#print("RandomForest score: %.4f (%.4f)" % (results.mean()*100, results.std()*100))
best_feature='X314'
todrop = features.loc[features['importance'] == 0].index
new_train_df = train_df.drop(todrop, 1)
new_train_df.head()
new_train_df.shape

new_test_df = test_df.drop(todrop, 1)
model_rfr = RandomForestRegressor(n_estimators=600, max_depth=3, min_samples_split=4, min_samples_leaf=60)
#results = cross_val_score(model_rfr, new_train_df, train_y, cv=10)
#print("RandomForest score: %.4f (%.4f)" % (results.mean()*100, results.std()*100))
#print(results)
model_svr = SVR(kernel='rbf',gamma=0.005, C=10, epsilon=5.0)

'''
results = cross_val_score(model_svr, train_df, train_y, cv=10)
print("SVR score: %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)

results = cross_val_score(model_svr, new_train_df, train_y, cv=10)
print("SVR score (only on most important features): %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)
'''
model_gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.007, max_depth=3, min_samples_split=6, 
                                      min_samples_leaf=60)
#results = cross_val_score(model_gbr, train_df, train_y, cv=10)
#print("GBR score: %.4f (%.4f)" % (results.mean()*100, results.std()*100))
#print(results)

#results = cross_val_score(model_gbr, new_train_df, train_y, cv=10)
#print("GBR score (imp features): %.4f (%.4f)" % (results.mean()*100, results.std()*100))
#print(results)
'''
importances = model_gbr.fit(new_train_df, train_y).feature_importances_
features = pd.DataFrame()
features['feature'] = new_train_df.columns
features['importance'] = importances

features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(12,24))
'''
'''
todrop = features.loc[features['importance'] == 0].index
new_train_df2 = new_train_df.drop(todrop, 1)
new_train_df2.head()
new_train_df2.shape
'''
model_gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.007, max_depth=3, min_samples_split=6, 
                                      min_samples_leaf=60)
'''
results = cross_val_score(model_gbr, new_train_df, train_y, cv=10)
print("GBR score: %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)

results = cross_val_score(model_gbr, new_train_df2, train_y, cv=10)
print("GBR score (imp features): %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)
'''
model_mlp = MLPRegressor(max_iter=200, solver='adam', learning_rate="constant")

'''
results = cross_val_score(model_mlp, train_df, train_y, cv=10)
print("MLP score: %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)

results = cross_val_score(model_mlp, new_train_df, train_y, cv=10)
print("MLP score: %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)
'''
model_ex = ExtraTreesRegressor(n_estimators=700, max_depth=3, min_samples_split=24, min_samples_leaf=5, bootstrap=True, oob_score=True)

'''
results = cross_val_score(model_ex, train_df, train_y, cv=10)
print("EXR score: %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)

results = cross_val_score(model_ex, new_train_df, train_y, cv=10)
print("EXR score (only on most important features): %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)
'''
model_ada = AdaBoostRegressor(n_estimators=50, learning_rate=0.01)

'''
results = cross_val_score(model_ada, train_df, train_y, cv=10)
print("ADA score: %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)

results = cross_val_score(model_ada, new_train_df, train_y, cv=10)
print("ADA score (only on most important features): %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)
'''
model_xgb = xgb.XGBRegressor(objective='reg:linear', n_estimators=50, max_depth=3, learning_rate=0.1, min_child_weight=30, subsample=0.9, colsample_bytree=0.7, reg_alpha=0.01)

'''
results = cross_val_score(model_xgb, train_df, train_y, cv=10)
print("XGB score: %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)

results = cross_val_score(model_xgb, new_train_df, train_y, cv=10)
print("XGB score (only on most important features): %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)
'''
model_sgd = SGDRegressor(alpha=0.02, penalty='l1', n_iter=10, power_t=0.2, average=False)

'''
results = cross_val_score(model_sgd, train_df, train_y, cv=10)
print("SGD score: %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)

results = cross_val_score(model_sgd, new_train_df, train_y, cv=10)
print("SGD score (only on most important features): %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)
'''
'''
    This code was borrowed and adapted
'''
class Stacking(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
    def fit_predict(self, X, X2, y, T, T2):
        X = np.array(X)
        X2 = np.array(X2)
        y = np.array(y)
        T = np.array(T)
        T2 = np.array(T2)
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=2016)
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        
        for i, clf in enumerate(self.base_models):
            print("Predicting with: ", clf[0])
            S_test_i = np.zeros((T.shape[0], self.n_folds))
            for j, (train_idx, test_idx) in enumerate(folds.split(X)):
                y_train = y[train_idx]  
                if clf[1] == 1:
                    X_train=X2[train_idx]
                    X_holdout = X2[test_idx]
                else:
                    X_train=X[train_idx]
                    X_holdout = X[test_idx]
                        
                clf[0].fit(X_train, y_train)
                y_pred = clf[0].predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                if clf[1] == 1:
                    S_test_i[:, j] = clf[0].predict(T2)[:]
                else:
                    S_test_i[:, j] = clf[0].predict(T)[:]        
            S_test[:, i] = S_test_i.mean(1)
            
        #self.stacker.fit(S_train, y)
        #y_pred = self.stacker.predict(S_test)[:]
        
        return S_train, S_test
base_models=[(model_rfr, 0), (model_gbr, 0), (model_xgb, 1), (model_ada, 0), (model_ex, 0)]

ens = Stacking(n_folds=10, stacker=model_gbr, base_models=base_models)
s_train, s_test = ens.fit_predict(train_df, new_train_df, train_y, test_df, new_test_df)
new_train = pd.DataFrame({
        "rfr": s_train[:, 0],
        "gbr": s_train[:, 1],
        "xgb": s_train[:, 2],
        "ada": s_train[:, 3],
        "ex": s_train[:, 4],
        "y": train_y
    })
new_train.to_csv('new_train.csv', index=False)
new_test = pd.DataFrame({
        "rfr": s_test[:, 0],
        "gbr": s_test[:, 1],
        "xgb": s_test[:, 2],
        "ada": s_test[:, 3],
        "ex": s_test[:, 4]
    })
new_test.to_csv('new_test.csv', index=False)

stacker = ElasticNet(normalize=True)
base_models=[(model_rfr, 0), (model_gbr, 0), (model_xgb, 1), (model_ada, 0), (model_ex, 0)]

ens = Stacking(n_folds=10, stacker=model_gbr, base_models=base_models)

y_pred=ens.fit_predict(train_df, new_train_df, train_y, test_df, new_test_df)


submission = pd.DataFrame({
        "ID": test_id,
        "y": y_pred
    })
submission.to_csv('mercedes_ens_opt.csv', index=False)

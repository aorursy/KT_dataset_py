# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/santander-value-prediction-challenge"))

# Any results you write to the current directory are saved as output.
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

import lightgbm as lgb

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
            
def get_oof(clf, x_train, y, x_test):
    #trainの一部を抜いて時の予測の平均を特徴量に加える。
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train.iloc[train_index]
        y_tr = y[train_index]
        x_te = x_train.iloc[test_index]

        clf.train(x_tr, y_tr)
        
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)




train_df = pd.read_csv('../input/santander-value-prediction-challenge/train.csv')
test_df = pd.read_csv('../input/santander-value-prediction-challenge/test.csv')
X_train = train_df.drop(["ID", "target"], axis=1)

y_train = np.log1p(train_df["target"].values)

X_test = test_df.drop(["ID"], axis=1)

ntrain = len(X_train)
ntest = len(X_test)
colsToRemove = []
for col in X_train.columns:
    if X_train[col].std() == 0: 
        colsToRemove.append(col)
        
# remove constant columns in the training set
X_train.drop(colsToRemove, axis=1, inplace=True)

# remove constant columns in the test set
X_test.drop(colsToRemove, axis=1, inplace=True) 

colsToRemove = []
colsScaned = []
dupList = {}

columns = X_train.columns

for i in range(len(columns)-1):
    v = X_train[columns[i]].values
    dupCols = []
    for j in range(i+1,len(columns)):
        if np.array_equal(v, X_train[columns[j]].values):
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j]) 
                colsScaned.append(columns[j])
                dupList[columns[i]] = dupCols
                
# remove duplicate columns in the training set
X_train.drop(colsToRemove, axis=1, inplace=True) 

# remove duplicate columns in the testing set
X_test.drop(colsToRemove, axis=1, inplace=True)
weight = ((X_train != 0).sum()/len(X_train)).values

tmp_train = X_train[X_train!=0].fillna(0)
tmp_test = X_test[X_test!=0].fillna(0)

X_train["weight_count"] = (tmp_train*weight).sum(axis=1)
X_test["weight_count"] = (tmp_test*weight).sum(axis=1)

X_train["count_not0"] = (X_train != 0).sum(axis=1)
X_test["count_not0"] = (X_test != 0).sum(axis=1)

X_train["sum"] = X_train.sum(axis=1)
X_test["sum"] = X_test.sum(axis=1)

X_train["var"] = X_train.var(axis=1)
X_test["var"] = X_test.var(axis=1)

X_train["mean"] = X_train.mean(axis=1)
X_test["mean"] = X_test.mean(axis=1)

X_train["std"] = X_train.std(axis=1)
X_test["std"] = X_test.std(axis=1)

X_train["max"] = X_train.max(axis=1)
X_test["max"] = X_test.max(axis=1)

X_train["min"] = X_train.min(axis=1)
X_test["min"] = X_test.min(axis=1)

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.cross_validation import KFold


SEED = 5
NFOLDS = 5

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)
ridge_params = {'alpha':30.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}

#Ridge oof method from Faron's kernel
#I was using this to analyze my vectorization, but figured it would be interesting to add the results back into the dataset
#It doesn't really add much to the score, but it does help lightgbm converge faster
ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, X_train, y_train, X_test)


rms = sqrt(mean_squared_error(y_train, ridge_oof_train))
print('Ridge OOF RMSE: {}'.format(rms))

print("Modeling Stage")

X_train["ridge_preds"] = ridge_oof_train
X_test["rigde_preds"] =  ridge_oof_test
dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 40,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)
    
    pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best_iteration))
    return pred_test_y, model, evals_result
pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)
print("LightGBM Training Completed...")
sub = pd.read_csv('../input/santander-value-prediction-challenge/sample_submission.csv')
sub["target"] = pred_test
print(sub.head())

sub.to_csv('sub_lgbm_ridge_agg.csv', index=False)

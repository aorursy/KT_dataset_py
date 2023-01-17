# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegressionCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', index_col=False)

test = pd.read_csv('../input/test.csv', index_col=False)

id_test = test['id']

test = test.drop(columns=['id'])

y = train['target']

X = train.drop(columns=['id', 'target'])
X.head()
plt.title('Visualize Target datasets')

sns.countplot(y)
train.isnull().sum().sort_values(ascending=False), test.isnull().sum().sort_values(ascending=False)
(train.nunique()/len(train.columns)).sort_values(ascending=True)
sns.pairplot(train.iloc[:, :4], hue='target')
pca = PCA(n_components=250)

X = pd.DataFrame(pca.fit_transform(X))

test = pd.DataFrame(pca.fit_transform(test))
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



clf = LogisticRegressionCV(cv=5)

clf = RandomForestClassifier()

clf = SVC(C=13.0, probability=True)

clf.fit(X_train, y_train)
print("Accuracy: ({:.2}, {:.2})".format(roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1]), roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])))



submit = pd.DataFrame()

submit['id'] = id_test

submit['target'] = clf.predict_proba(test)[:, 1]

submit.to_csv('submit.csv', index=False)
#train.min().sort_values(ascending=False)
#train.max().sort_values(ascending=True)




params = {'learning_rate': 0.3,

              'application': 'binary',

              'num_leaves': 31,

              'verbosity': -1,

              'metric': 'auc',

              'data_random_seed': 2,

              'bagging_fraction': 0.8,

              'feature_fraction': 0.6,

              'nthread': 4,

              'lambda_l1': 1,

              'lambda_l2': 1}

# train_data = lgb.Dataset(X_train, label=y_train)

# val_data = lgb.Dataset(X_test, label=y_test)

# watchlist = [train_data, val_data]



# model_lgb = lgb.train(params, train_set=train_data, valid_sets=watchlist)



from sklearn.model_selection import KFold, StratifiedKFold

N_FOLDS = 10

folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(X))

sub = np.zeros(len(test))

scores = [0 for _ in range(folds.n_splits)]

for fold_, (train_idx, val_idx) in enumerate(folds.split(X.values, y)):

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]

    X_val, y_val = X.loc[val_idx], y.loc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)

    val_data = lgb.Dataset(X_val, label=y_val)

    watchlist = [train_data, val_data]

    clf = lgb.train(params, train_set = train_data, valid_sets=watchlist)

    oof[val_idx] = clf.predict(X_val)

    sub += clf.predict(test)/folds.n_splits

    scores[fold_] = roc_auc_score(y[val_idx], oof[val_idx])

    print("Fold {}: {}".format(fold_+1, round(scores[fold_],5)))

    

print("CV score(auc): {:<8.5f}, (std: {:<8.5f})".format(roc_auc_score(y, oof), np.std(scores)))

submit = pd.DataFrame()

submit['id'] = id_test

submit['target'] = sub

submit.to_csv('submit_lgbcv.csv', index=False)
print(type(submit))

if(isinstance(submit,(pd.core.frame.DataFrame))):

    print("True")
params = {

    'learning_rate': 0.2,

    'application': 'binary',

    'num_leaves': 31,

    'verbosity': -1,

    'metric': 'auc',

    'data_random_seed': 2,

    'bagging_fraction': 0.8,

    'feature_fraction': 0.6,

    'nthread': 4,

    'lambda_l1': 1,

    'lambda_l2': 1}

def lightgbm(train, target, params, test_size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)

    val_data = lgb.Dataset(X_test, label=y_test)

    watchlist = [train_data, val_data]

    model_lgb = lgb.train(params, train_set=train_data, valid_sets=watchlist)

    return model_lgb

model = lightgbm(train=X, target=y, params=params)



params = {

    'learning_rate': 0.2,

    'application': 'binary',

    'num_boost_round': 100,

    'nfold': 5,

    'num_leaves': 31,

    'verbosity': -1,

    'metric': 'auc',

    'data_random_seed': 2,

    'bagging_fraction': 0.8,

    'feature_fraction': 0.6,

    'nthread': 4,

    'lambda_l1': 1,

    'lambda_l2': 1,

    'early_stopping_rounds': 40,

}

def lightgbmcv(train, target, params):

    train_data=lgb.Dataset(train, label=target)

    model_lgbcv = lgb.cv(params, train_set=train_data)

    return model_lgbcv

model = lightgbmcv(X, y, params=params)

lightgbmcv(X, y, params)
import xgboost as xgb

params = {

    'learning_rates': 0.2,

    'eta': 0.02, 

    'max_depth': 10, 

    'subsample': 0.7, 

    'colsample_bytree': 0.7, 

    'objective': 'binary:logistic', 

    'seed': 99, 

    'silent': 1, 

    'eval_metric':'auc', 

    'nthread':4}



def xgboost(train, target, early_stopping_rounds=10, test_size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=42)

    xgb_train = xgb.DMatrix(train, label=target)

    model = xgb.train(params, xgb_train, verbose_eval=1)

    return model

xgboost(X, y)



params = {

    'learning_rates': 0.2,

    'eta': 0.02, 

    'max_depth': 10, 

    'subsample': 0.7, 

    'colsample_bytree': 0.7, 

    'objective': 'binary:logistic', 

    'seed': 99, 

    'silent': 1, 

    'eval_metric':'auc', 

    'nthread':4}



def xgboostcv(train, target, nfold=5, early_stopping_rounds=10):

    xgb_train = xgb.DMatrix(train, label=target)

    model = xgb.cv(params, xgb_train, 5000, nfold=nfold, early_stopping_rounds=early_stopping_rounds, verbose_eval=1)

    return model

#xgboostcv(X, y)

xgboost(X, y)
print("Accuracy: ({:.2}, {:.2})".format(accuracy_score(y_train, np.round(model_lgb.predict(X_train))), accuracy_score(y_test, np.round(model_lgb.predict(X_test)))))
submit = pd.DataFrame()

submit['id'] = id_test

submit['target'] = model_lgb.predict(test)

submit.to_csv('submit_lgb.csv', index=False)


def cross_validation(train, target, test, model, folds, N_FOLDS=10):

    oof = np.zeros(len(train))

    sub = np.zeros(len(test))

    scores = [0 for _ in range(folds.n_splits)]

    for fold_, (train_index, val_index) in enumerate(folds.split(X, y)):

        if(isinstance(X,(pd.core.frame.DataFrame))):

            X_train, y_train = train.loc[train_index], y.loc[train_index]

            X_val, y_val = train.loc[val_index], y.loc[val_index]

        else:

            X_train, y_train = train[train_index], y[train_index]

            X_val, y_val = train[val_index], y[val_index]

        model.fit(X_train, y_train)

        oof[val_idx] = model.predict(X_val)

        sub += model.predict(test)/folds.n_splits

        scores[fold_] = roc_auc_score(y[val_index], oof[val_index])

        print("Fold {}: {}".format(fold_+1, round(scores[fold_],5)))

    print("CV score(auc): {:<8.5f}, (std: {:<8.5f})".format(roc_auc_score(y, oof), np.std(scores)))

folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

clf = LogisticRegression(C=13.0)

cross_validation(train=X, target=y, test=test, model=clf, folds=folds)    
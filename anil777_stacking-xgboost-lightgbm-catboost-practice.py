import pandas as pd

import numpy as np

from scipy.stats import skew

import xgboost as xgb

from sklearn.model_selection import KFold

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression

from math import sqrt

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier

import gc
NFOLDS = 3

SEED = 0

NROWS = None
data = pd.read_csv('../input/application_train.csv')

test = pd.read_csv('../input/application_test.csv')

prev = pd.read_csv('../input/previous_application.csv')

data.head()
categorical_feats = [f for f in data.columns if data[f].dtype == 'object']

print(categorical_feats)
for f_ in categorical_feats:

    data[f_], indexer = pd.factorize(data[f_])

    #print(data[f_])

    print(indexer)

    test[f_] = indexer.get_indexer(test[f_])

    #print(test[f_])
gc.enable()
y_train = data['TARGET']

del data['TARGET']
prev_cat_features = [f_ for f_ in prev.columns if prev[f_].dtype == 'object']

for f_ in prev_cat_features:

    prev[f_], _ = pd.factorize(prev[f_])

    
avg_prev = prev.groupby('SK_ID_CURR').mean()

cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()

avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']

del avg_prev['SK_ID_PREV']
x_train = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

x_test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
x_train = x_train.fillna(0)

x_test= x_test.fillna(0)



ntrain = x_train.shape[0]

ntest = x_test.shape[0]

print(ntrain)
excluded_feats = ['SK_ID_CURR']

features = [f_ for f_ in x_train.columns if f_ not in excluded_feats]



x_train = x_train[features]

x_test = x_test[features]
kf = KFold(n_splits = NFOLDS, shuffle=True, random_state=SEED)

class SklearnWrapper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict_proba(x)[:,1]
class CatboostWrapper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_seed'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict_proba(x)[:,1]

        

class LightGBMWrapper(object):

    def __init__(self, clf, seed=0, params=None):

        params['feature_fraction_seed'] = seed

        params['bagging_seed'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict_proba(x)[:,1]





class XgbWrapper(object):

    def __init__(self, seed=0, params=None):

        self.param = params

        self.param['seed'] = seed

        self.nrounds = params.pop('nrounds', 250)



    def train(self, x_train, y_train):

        dtrain = xgb.DMatrix(x_train, label=y_train)

        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)



    def predict(self, x):

        return self.gbdt.predict(xgb.DMatrix(x))

def get_oof(clf):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train.loc[train_index]

        y_tr = y_train.loc[train_index]

        x_te = x_train.loc[test_index]



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

et_params = {

    'n_jobs': 16,

    'n_estimators': 200,

    'max_features': 0.5,

    'max_depth': 12,

    'min_samples_leaf': 2,

}



rf_params = {

    'n_jobs': 16,

    'n_estimators': 200,

    'max_features': 0.2,

    'max_depth': 12,

    'min_samples_leaf': 2,

}



xgb_params = {

    'seed': 0,

    'colsample_bytree': 0.7,

    'silent': 1,

    'subsample': 0.7,

    'learning_rate': 0.075,

    'objective': 'binary:logistic',

    'max_depth': 4,

    'num_parallel_tree': 1,

    'min_child_weight': 1,

    'nrounds': 200

}



catboost_params = {

    'iterations': 200,

    'learning_rate': 0.5,

    'depth': 3,

    'l2_leaf_reg': 40,

    'bootstrap_type': 'Bernoulli',

    'subsample': 0.7,

    'scale_pos_weight': 5,

    'eval_metric': 'AUC',

    'od_type': 'Iter',

    'allow_writing_files': False

}



lightgbm_params = {

    'n_estimators':200,

    'learning_rate':0.1,

    'num_leaves':123,

    'colsample_bytree':0.8,

    'subsample':0.9,

    'max_depth':15,

    'reg_alpha':0.1,

    'reg_lambda':0.1,

    'min_split_gain':0.01,

    'min_child_weight':2    

}
xg = XgbWrapper(seed=SEED, params=xgb_params)

et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

cb = CatboostWrapper(clf= CatBoostClassifier, seed = SEED, params=catboost_params)

lg = LightGBMWrapper(clf = LGBMClassifier, seed = SEED, params = lightgbm_params)
xg_oof_train, xg_oof_test = get_oof(xg)

et_oof_train, et_oof_test = get_oof(et)

rf_oof_train, rf_oof_test = get_oof(rf)

cb_oof_train, cb_oof_test = get_oof(cb)
print("XG-CV: {}".format(sqrt(mean_squared_error(y_train, xg_oof_train))))

print("ET-CV: {}".format(sqrt(mean_squared_error(y_train, et_oof_train))))

print("RF-CV: {}".format(sqrt(mean_squared_error(y_train, rf_oof_train))))

print("RF-CV: {}".format(sqrt(mean_squared_error(y_train, cb_oof_train))))
x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, cb_oof_train), axis=1)

x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, cb_oof_test), axis=1)

print("{},{}".format(x_train.shape, x_test.shape))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')

train_set.shape, test_set.shape
train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')

train_set.shape, test_set.shape
#y_train = train_set['loss'].ravel()

y_train = np.log(train_set['loss'].ravel())

train_set['loss'].describe()
train = train_set.drop(['id','loss'], axis=1)

test = test_set.drop(['id'], axis=1)

train.shape, test.shape
train_test = pd.concat((train,test)).reset_index(drop=True)

print(train_test.shape)
features = train.columns

cats = [feat for feat in features if 'cat' in feat]

len(cats)
print(train_test['cat1'].value_counts())
for feat in cats:

    train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]

    

print(train_test['cat1'].value_counts())
conts = [feat for feat in features if 'cont' in feat]

print(conts)
train_test[conts].describe()
ntrain = train.shape[0]

x_train = np.array(train_test.iloc[:ntrain,:])

x_test = np.array(train_test.iloc[ntrain:, :])
from sklearn.cross_validation import KFold

seed = 0

NFOLDS = 4

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=seed)
import xgboost as xgb

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



class SklearnWrapper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)





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



    for i, (train_index, test_index) in enumerate(kf):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
et_params = {

    'n_jobs': 16,

    'n_estimators': 100,

    'max_features': 0.5,

    'max_depth': 12,

    'min_samples_leaf': 2,

}



rf_params = {

    'n_jobs': 16,

    'n_estimators': 100,

    'max_features': 0.2,

    'max_depth': 8,

    'min_samples_leaf': 2,

}



xgb_params = {

    'seed': 0,

    'colsample_bytree': 0.7,

    'silent': 1,

    'subsample': 0.7,

    'learning_rate': 0.075,

    'objective': 'reg:linear',

    'max_depth': 7,

    'num_parallel_tree': 1,

    'min_child_weight': 1,

    'eval_metric': 'mae',

    'nrounds': 350

}
ntest = test.shape[0]

xg = XgbWrapper(seed=seed, params=xgb_params)

et = SklearnWrapper(clf=ExtraTreesRegressor, seed=seed, params=et_params)

rf = SklearnWrapper(clf=RandomForestRegressor, seed=seed, params=rf_params)



xg_oof_train, xg_oof_test = get_oof(xg)

et_oof_train, et_oof_test = get_oof(et)

rf_oof_train, rf_oof_test = get_oof(rf)



print("XG-CV: {}".format(mean_absolute_error(y_train, xg_oof_train)))

print("ET-CV: {}".format(mean_absolute_error(y_train, et_oof_train)))

print("RF-CV: {}".format(mean_absolute_error(y_train, rf_oof_train)))
x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train), axis=1)

x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test), axis=1)

print(x_train.shape, x_test.shape)
x_train
dtrain = xgb.DMatrix(x_train, label=y_train)

dtest = xgb.DMatrix(x_test)



xgb_params = {

    'seed': 0,

    'colsample_bytree': 0.8,

    'silent': 1,

    'subsample': 0.6,

    'learning_rate': 0.01,

    'objective': 'reg:linear',

    'max_depth': 4,

    'num_parallel_tree': 1,

    'min_child_weight': 1,

    'eval_metric': 'mae',

}



res = xgb.cv(xgb_params, dtrain, num_boost_round=500, nfold=4, seed=seed, stratified=False,

             early_stopping_rounds=25, verbose_eval=10, show_stdv=True)
best_nrounds = res.shape[0] - 1

cv_mean = res.iloc[-1, 0]

cv_std = res.iloc[-1, 1]



print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
gbdt = xgb.train(xgb_params, dtrain, best_nrounds)
submission = pd.read_csv("../input/sample_submission.csv")

submission.iloc[:, 1] = np.exp(gbdt.predict(dtest))

submission.to_csv('xgstacker_starter_log.sub.csv', index=None)
print(check_output(["ls", "."]).decode("utf8"))
submission.head()
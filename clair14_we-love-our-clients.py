# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
train = pd.read_csv('../input/train.csv', index_col='ID')
test = pd.read_csv('../input/test.csv', index_col='ID')
train.head()
train.info()
sns.distplot(train.target)
sns.distplot(np.log1p(train.target))
y = np.log1p(train.target)
def rmsle(y, pred):
    assert len(y) == len(pred)
    return np.sqrt(np.mean(np.power(y-pred, 2)))
cols_with_onlyone_val = train.columns[train.nunique() == 1]
train.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
test.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
train = train.round(32)
test = test.round(32)
colsToRemove = []
columns = train.columns
for i in range(len(columns)-1):
    v = train[columns[i]].values
    dupCols = []
    for j in range(i + 1,len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            colsToRemove.append(columns[j])
train.drop(colsToRemove, axis=1, inplace=True)
test.drop(colsToRemove, axis=1, inplace=True)
X = train.drop('target', axis = 1)
X['mean'] = X.mean(axis = 1)
X['sum'] = X.sum(axis = 1)
X['max'] = X.max(axis = 1)
X['min'] = X.min(axis = 1)
X['std'] = X.std(axis = 1)
test['mean'] = test.mean(axis = 1)
test['sum'] = test.sum(axis = 1)
test['max'] = test.max(axis = 1)
test['min'] = test.min(axis = 1)
test['std'] = test.std(axis = 1)
cols = X[X > 0].count()[X[X > 0].count() > 200].sort_values().index
X_train = X[cols].copy()
X_test = test[cols].copy()
clust = KMeans(n_clusters=15).fit(X_train)
X_with_clust = X_train.join(pd.DataFrame(clust.transform(X_train), columns= [ 'clust_' + str(i) for i in range(15)], index=X_train.index))
X_with_clust['cluster_name'] = clust.predict(X_train)
test_with_clust = X_test.join(pd.DataFrame(clust.transform(X_test), columns= [ 'clust_' + str(i) for i in range(15)], index=X_test.index))
test_with_clust['cluster_name'] = clust.predict(X_test)
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_with_clust, y)
features = pd.DataFrame({'importance': rf.feature_importances_, 'name': X_with_clust.columns})
X_short = X_with_clust[features.sort_values('importance', ascending=False).name[:500]]
test_short = test_with_clust[features.sort_values('importance', ascending=False).name[:500]]
import lightgbm
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

N_HYPEROPT_PROBES = 500

HYPEROPT_ALGO = tpe.suggest  #  tpe.suggest OR hyperopt.rand.suggest
def get_lgb_params(space):
    lgb_params = dict()
    lgb_params['boosting_type'] = space['boosting_type'] if 'boosting_type' in space else 'gbdt'
    lgb_params['application'] = 'regression'
    lgb_params['metric'] = 'l2_root'
    lgb_params['learning_rate'] = space['learning_rate']
    lgb_params['num_leaves'] = int(space['num_leaves'])
    lgb_params['min_data_in_leaf'] = int(space['min_data_in_leaf'])
    lgb_params['min_sum_hessian_in_leaf'] = space['min_sum_hessian_in_leaf']
    lgb_params['max_depth'] = -1
    lgb_params['lambda_l1'] = space['lambda_l1'] if 'lambda_l1' in space else 0.0
    lgb_params['lambda_l2'] = space['lambda_l2'] if 'lambda_l2' in space else 0.0
    lgb_params['max_bin'] = int(space['max_bin']) if 'max_bin' in space else 256
    lgb_params['feature_fraction'] = space['feature_fraction']
    lgb_params['bagging_fraction'] = space['bagging_fraction']
    lgb_params['bagging_freq'] = int(space['bagging_freq']) if 'bagging_freq' in space else 1

    return lgb_params
obj_call_count = 0
cur_best_loss = np.inf
X_1, X_test, y_1, y_test = train_test_split(X_short, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_1, y_1, test_size=0.3)
D_train = lightgbm.Dataset(X_train, y_train)
D_val = lightgbm.Dataset(X_val, y_val)
def objective(space):
    global obj_call_count, cur_best_loss

    obj_call_count += 1

    #print('\nXGB objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    lgb_params = get_lgb_params(space)

    sorted_params = sorted(space.items(), key=lambda z: z[0])
    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    #print('Params: {}'.format(params_str) )

    model = lightgbm.train(lgb_params,
                           D_train,
                           num_boost_round=10000,
                           # metrics='mlogloss',
                           valid_sets=D_val,
                           # valid_names='val',
                           # fobj=None,
                           # feval=None,
                           # init_model=None,
                           # feature_name='auto',
                           # categorical_feature='auto',
                           early_stopping_rounds=100,
                           # evals_result=None,
                           verbose_eval=False,
                           # learning_rates=None,
                           # keep_training_booster=False,
                           # callbacks=None
                           )

    nb_trees = model.best_iteration
    val_loss = model.best_score

    #print('nb_trees={} val_loss={}'.format(nb_trees, val_loss))

    y_pred = model.predict(X_test, num_iteration=nb_trees)
    test_loss = rmsle(y_test, y_pred)

    #print('test_loss={}'.format(test_loss))

    if test_loss<cur_best_loss:
        cur_best_loss = test_loss
        print('NEW BEST LOSS={}'.format(cur_best_loss))


    return{'loss':test_loss, 'status': STATUS_OK }
space ={
        'num_leaves': hp.quniform ('num_leaves', 10, 200, 1),
        'min_data_in_leaf':  hp.quniform ('min_data_in_leaf', 10, 200, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -5.0, -2.3),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        'max_bin': hp.quniform ('max_bin', 64, 512, 1),
        'bagging_freq': hp.quniform ('bagging_freq', 1, 5, 1),
        'lambda_l1': hp.uniform('lambda_l1', 0, 10 ),
        'lambda_l2': hp.uniform('lambda_l2', 0, 10 ),
       }
trials = Trials()
best = hyperopt.fmin(fn=objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=0)
print(get_lgb_params(best))
def run_lgb(train_X, train_y, val_X, val_y, test_X, params):
    
    lgtrain = lightgbm.Dataset(train_X, label=train_y)
    lgval = lightgbm.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lightgbm.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=False, evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result
X_train = X_short.reset_index(drop=True)
y_train = y.reset_index(drop=True)
params = get_lgb_params(best)
kf = KFold(n_splits=5, shuffle=True, random_state=2017)
pred_test_lgb = 0
for dev_index, val_index in kf.split(X_train):
    dev_X, val_X = X_train.loc[dev_index,:], X_train.loc[val_index,:]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_short,params)
    pred_test_lgb += pred_test
pred_test_lgb /= 5.
pred_test_lgb = np.expm1(pred_test_lgb)
from catboost import CatBoostRegressor
cb_model = CatBoostRegressor(iterations=500,
                             learning_rate=0.05,
                             depth=10,
                             eval_metric='RMSE',
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)
X_train, X_val, y_train, y_val = train_test_split(X_short, y, test_size = 0.2, random_state = 42)
cb_model.fit(X_train, y_train,
             eval_set=(X_val, y_val),
             use_best_model=True,
             verbose=False)
pred_test_cat = np.expm1(cb_model.predict(test_short))
sub = pd.DataFrame()
sub['ID'] = test.index
sub['target'] = (pred_test_lgb + pred_test_cat)/2
sub.to_csv('submission.csv',index=False)

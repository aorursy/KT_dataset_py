import numpy as np

import pandas as pd



from hyperopt import hp, tpe

from hyperopt.fmin import fmin



from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer



import xgboost as xgb



import lightgbm as lgbm
df = pd.read_csv('../input/train.csv')



X = df.drop(['id', 'target'], axis=1)

Y = df['target']
def gini(truth, predictions):

    g = np.asarray(np.c_[truth, predictions, np.arange(len(truth)) ], dtype=np.float)

    g = g[np.lexsort((g[:,2], -1*g[:,1]))]

    gs = g[:,0].cumsum().sum() / g[:,0].sum()

    gs -= (len(truth) + 1) / 2.

    return gs / len(truth)



def gini_xgb(predictions, truth):

    truth = truth.get_label()

    return 'gini', -1.0 * gini(truth, predictions) / gini(truth, truth)



def gini_lgb(truth, predictions):

    score = gini(truth, predictions) / gini(truth, truth)

    return 'gini', score, True



def gini_sklearn(truth, predictions):

    return gini(truth, predictions) / gini(truth, truth)



gini_scorer = make_scorer(gini_sklearn, greater_is_better=True, needs_proba=True)
def objective(params):

    params = {'n_estimators': int(params['n_estimators']), 'max_depth': int(params['max_depth'])}

    clf = RandomForestClassifier(n_jobs=4, class_weight='balanced', **params)

    score = cross_val_score(clf, X, Y, scoring=gini_scorer, cv=StratifiedKFold()).mean()

    print("Gini {:.3f} params {}".format(score, params))

    return score



space = {

    'n_estimators': hp.quniform('n_estimators', 25, 500, 25),

    'max_depth': hp.quniform('max_depth', 1, 10, 1)

}



best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=10)
print("Hyperopt estimated optimum {}".format(best))
def objective(params):

    params = {

        'max_depth': int(params['max_depth']),

        'gamma': "{:.3f}".format(params['gamma']),

        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),

    }

    

    clf = xgb.XGBClassifier(

        n_estimators=250,

        learning_rate=0.05,

        n_jobs=4,

        **params

    )

    

    score = cross_val_score(clf, X, Y, scoring=gini_scorer, cv=StratifiedKFold()).mean()

    print("Gini {:.3f} params {}".format(score, params))

    return score



space = {

    'max_depth': hp.quniform('max_depth', 2, 8, 1),

    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),

    'gamma': hp.uniform('gamma', 0.0, 0.5),

}



best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=10)
print("Hyperopt estimated optimum {}".format(best))
def objective(params):

    params = {

        'num_leaves': int(params['num_leaves']),

        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),

    }

    

    clf = lgbm.LGBMClassifier(

        n_estimators=500,

        learning_rate=0.01,

        **params

    )

    

    score = cross_val_score(clf, X, Y, scoring=gini_scorer, cv=StratifiedKFold()).mean()

    print("Gini {:.3f} params {}".format(score, params))

    return score



space = {

    'num_leaves': hp.quniform('num_leaves', 8, 128, 2),

    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),

}



best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=10)
print("Hyperopt estimated optimum {}".format(best))
rf_model = RandomForestClassifier(

    n_jobs=4,

    class_weight='balanced',

    n_estimators=325,

    max_depth=5

)



xgb_model = xgb.XGBClassifier(

    n_estimators=250,

    learning_rate=0.05,

    n_jobs=4,

    max_depth=2,

    colsample_bytree=0.7,

    gamma=0.15

)



lgbm_model = lgbm.LGBMClassifier(

    n_estimators=500,

    learning_rate=0.01,

    num_leaves=16,

    colsample_bytree=0.7

)



models = [

    ('Random Forest', rf_model),

    ('XGBoost', xgb_model),

    ('LightGBM', lgbm_model),

]



for label, model in models:

    scores = cross_val_score(model, X, Y, cv=StratifiedKFold(), scoring=gini_scorer)

    print("Gini coefficient: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))
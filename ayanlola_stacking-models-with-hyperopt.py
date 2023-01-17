import os

import random

import warnings



import numpy as np

import pandas as pd

from hyperopt import hp, tpe

from hyperopt.fmin import fmin

from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold

from mlxtend.classifier import StackingCVClassifier



from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import make_scorer

warnings.simplefilter(action='ignore', category=FutureWarning)



random_state = 1

random.seed(random_state)

np.random.seed(random_state)

os.environ['PYTHONHASHSEED'] = str(random_state)





print('> Loading data')

X_train = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col='Id')

X_test = pd.read_csv('/kaggle/input/learn-together/test.csv', index_col='Id')



y_train = X_train['Cover_Type'].copy()

X_train = X_train.drop(['Cover_Type'], axis='columns')





print('> Adding and dropping features')



def add_features(X_):

    X = X_.copy()



    X['Hydro_Elevation_diff'] = X[['Elevation',

                                   'Vertical_Distance_To_Hydrology']

                                  ].diff(axis='columns').iloc[:, [1]]



    X['Hydro_Euclidean'] = np.sqrt(X['Horizontal_Distance_To_Hydrology']**2 +

                                   X['Vertical_Distance_To_Hydrology']**2)



    X['Hydro_Fire_sum'] = X[['Horizontal_Distance_To_Hydrology',

                             'Horizontal_Distance_To_Fire_Points']

                            ].sum(axis='columns')



    X['Hydro_Fire_diff'] = X[['Horizontal_Distance_To_Hydrology',

                              'Horizontal_Distance_To_Fire_Points']

                             ].diff(axis='columns').iloc[:, [1]].abs()



    X['Hydro_Road_sum'] = X[['Horizontal_Distance_To_Hydrology',

                             'Horizontal_Distance_To_Roadways']

                            ].sum(axis='columns')



    X['Hydro_Road_diff'] = X[['Horizontal_Distance_To_Hydrology',

                              'Horizontal_Distance_To_Roadways']

                             ].diff(axis='columns').iloc[:, [1]].abs()



    X['Road_Fire_sum'] = X[['Horizontal_Distance_To_Roadways',

                            'Horizontal_Distance_To_Fire_Points']

                           ].sum(axis='columns')



    X['Road_Fire_diff'] = X[['Horizontal_Distance_To_Roadways',

                             'Horizontal_Distance_To_Fire_Points']

                            ].diff(axis='columns').iloc[:, [1]].abs()

    

    # Compute Soil_Type number from Soil_Type binary columns

    X['Stoneyness'] = sum(i * X['Soil_Type{}'.format(i)] for i in range(1, 41))

    

    # For all 40 Soil_Types, 1=rubbly, 2=stony, 3=very stony, 4=extremely stony, 0=?

    stoneyness = [4, 3, 1, 1, 1, 2, 0, 0, 3, 1, 

                  1, 2, 1, 0, 0, 0, 0, 3, 0, 0, 

                  0, 4, 0, 4, 4, 3, 4, 4, 4, 4, 

                  4, 4, 4, 4, 1, 4, 4, 4, 4, 4]

    

    # Replace Soil_Type number with "stoneyness" value

    X['Stoneyness'] = X['Stoneyness'].replace(range(1, 41), stoneyness)

    

    return X





def drop_features(X_):

    X = X_.copy()

    drop_cols = ['Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type14', 'Soil_Type15', 

                 'Soil_Type16', 'Soil_Type18', 'Soil_Type19', 'Soil_Type21', 'Soil_Type25', 

                 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type34', 'Soil_Type36', 

                 'Soil_Type37']

    

    X = X.drop(drop_cols, axis='columns')



    return X



print('  -- Processing train data')

X_train = add_features(X_train)

X_train = drop_features(X_train)



print('  -- Processing test data')

X_test = add_features(X_test)

X_test = drop_features(X_test)
print('> Adding cluster based feature')

from sklearn.mixture import GaussianMixture



gmix = GaussianMixture(n_components=10)

gmix.fit(X_test)



X_train['Test_Cluster'] = gmix.predict(X_train)

X_test['Test_Cluster'] = gmix.predict(X_test)
#gini evaluation metrics

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
#Tuning Random Forest with hyperopt library

def objective(params):

    params = {'n_estimators': int(params['n_estimators']), 'max_depth': int(params['max_depth'])}

    clf = RandomForestClassifier(n_jobs=4, class_weight='balanced', **params)

    score = cross_val_score(clf, X_train, y_train, scoring=gini_scorer, cv=StratifiedKFold()).mean()

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

#optimum parameter for RF

print("Hyperopt estimated optimum {}".format(best))
#Tuning XGBoost

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

#Tree complexity (max_depth)

#Gamma - Make individual trees conservative, reduce overfitting

#Column sample per tree - reduce overfitting

def objective(params):

    params = {

        'max_depth': int(params['max_depth']),

        'gamma': "{:.3f}".format(params['gamma']),

        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),

    }

    

    clf =XGBClassifier(

        n_estimators=250,

        learning_rate=0.05,

        n_jobs=4,

        **params

    )

    

    score = cross_val_score(clf, X_train, y_train, scoring=gini_scorer, cv=StratifiedKFold()).mean()

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
#optimum parameters for XG

print("Hyperopt estimated optimum {}".format(best))
#Tuning LightGBM

import lightgbm as lgbm

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

    

    score = cross_val_score(clf, X_train, y_train, scoring=gini_scorer, cv=StratifiedKFold()).mean()

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
#optimum for LGBM

print("Hyperopt estimated optimum {}".format(best))
#Hyperopt estimated optimum_RF {'max_depth': 10.0, 'n_estimators': 250.0}

#Hyperopt estimated optimum_LGBM {'colsample_bytree': 0.911426972757106, 'num_leaves': 88.0}

#Hyperopt estimated optimum_Xg {'colsample_bytree': 0.8284142875897897, 'gamma': 0.17464593604973394, 'max_depth': 8.0}

print('> Setting up classifiers')

n_jobs = -1



#ab_clf = AdaBoostClassifier(n_estimators=200,

 #                           base_estimator=DecisionTreeClassifier(

  #                              min_samples_leaf=2,

  #                              random_state=random_state),

  #                          random_state=random_state)



lg_clf = LGBMClassifier(n_estimators=400,

                        num_leaves=88,

                        verbosity=0,

                        colsample_bytree=0.91,

                        random_state=random_state,

                        n_jobs=n_jobs)



rf_clf = RandomForestClassifier(n_estimators=250,

                                max_depth=10,

                                min_samples_leaf=1,

                                verbose=0,

                                random_state=random_state,

                                n_jobs=n_jobs)





xgb_clf = xgb.XGBClassifier(

    n_estimators=250,

    learning_rate=0.05,

    n_jobs=4,

    max_depth=8,

    colsample_bytree=0.83,

    gamma=0.17

)



ensemble = [('xgb', xgb_clf),

            ('lg', lg_clf),

            ('rf', rf_clf)]









stack = StackingCVClassifier(classifiers=[clf for label, clf in ensemble],

                             meta_classifier=xgb_clf,

                             cv=5,

                             use_probas=True,

                             use_features_in_secondary=True,

                             verbose=1,

                             random_state=random_state,

                             n_jobs=n_jobs)





print('> Fitting & predicting')

stack = stack.fit(X_train.as_matrix(), y_train.as_matrix())

prediction = stack.predict(X_test.as_matrix())
print('> Creating submission')

submission = pd.DataFrame({'Id': X_test.index, 'Cover_Type': prediction})

submission.to_csv('submission_tunde.csv', index=False)





print('> Done !')
import pandas as pd

from skopt import BayesSearchCV

from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier



train = pd.read_csv('../input/train.csv').drop("ID_code", axis='columns')

targets = train['target']

train.drop('target', axis='columns', inplace=True)



X_train, X_test, y_train, y_test = train_test_split(train, targets, train_size=0.80, random_state=0)



opt = BayesSearchCV(

    LGBMClassifier(objective='binary', metric='auc', n_estimators=20000, bagging_fraction=0.05),

    {

        'num_leaves': (13, 31),

        'boost': ['gbdt', 'goss'],

        'learning_rate': (0.008, 0.01),

        'max_depth': (8, 16),

        'feature_fraction': (0.05, 0.10, 0.25),

    },

     fit_params={

             'eval_set': (X_test, y_test),

             'eval_metric': 'auc', 

             'early_stopping_rounds': 1000,

             },

    n_jobs=16, cv=2,

)



opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)

print("test score: %s" % opt.score(X_test, y_test))

print("Best parameters: ", opt.best_params_)
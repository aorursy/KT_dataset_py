import pandas as pd

import numpy as np

import lightgbm as lgb

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

import optuna
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/gender_submission.csv')
# SexとEmbarkedのOne-Hotエンコーディング

train = pd.get_dummies(train, columns=['Sex', 'Embarked'])

test = pd.get_dummies(test, columns=['Sex', 'Embarked'])



# 不要な列の削除

train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)



# trainの表示

display(train.head())



X_train = train.drop(['Survived'], axis=1)  # X_trainはtrainのSurvived列以外

Y_train = train['Survived']  # Y_trainはtrainのSurvived列
def objective(trial):

    kf = KFold(n_splits=3)

    gbm = lgb.LGBMClassifier(objective='binary')

    oof = np.zeros(len(train))



    for fold, (train_index, valid_index) in enumerate(kf.split(X_train, Y_train)):

        train_x, valid_x = X_train.iloc[train_index], X_train.iloc[valid_index]

        train_y, valid_y  = Y_train[train_index], Y_train[valid_index]

        gbm = lgb.LGBMClassifier(objective='binary',

                                 reg_alpha=trial.suggest_loguniform('reg_alpha', 1e-4, 100.0),

                                 reg_lambda=trial.suggest_loguniform('reg_lambda', 1e-4, 100.0),

                                 silent=True)

        gbm.fit(train_x, train_y, eval_set = [(valid_x, valid_y)],

                early_stopping_rounds=20,

                verbose= -1) # 学習の状況を表示しない

        oof[valid_index] = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)



    accuracy = accuracy_score(Y_train, oof)

    return 1.0 - accuracy

    
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

# シードを固定。

# 参考：https://qiita.com/phorizon20/items/1b795beb202c2dc378ed



study.optimize(objective, n_trials=100)
kf = KFold(n_splits=3)

oof = np.zeros(len(train))

for fold, (train_index, valid_index) in enumerate(kf.split(X_train, Y_train)):

    train_x, valid_x = X_train.iloc[train_index], X_train.iloc[valid_index]

    train_y, valid_y  = Y_train[train_index], Y_train[valid_index]



    gbm = lgb.LGBMClassifier(objective='binary')

    gbm.fit(train_x, train_y, eval_set = [(valid_x, valid_y)],

            early_stopping_rounds=20,

            verbose= -1) # 学習の状況を表示しない

    

    oof[valid_index] = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)



round(accuracy_score(Y_train, oof), 3)
kf = KFold(n_splits=3)

oof = np.zeros(len(train))

for fold, (train_index, valid_index) in enumerate(kf.split(X_train, Y_train)):

    train_x, valid_x = X_train.iloc[train_index], X_train.iloc[valid_index]

    train_y, valid_y  = Y_train[train_index], Y_train[valid_index]



    gbm = lgb.LGBMClassifier(objective='binary',

                            reg_alpha=0.19628224813442816,

                            reg_lambda=1.9549524484259886)

    gbm.fit(train_x, train_y, eval_set = [(valid_x, valid_y)],

            early_stopping_rounds=20,

            verbose= -1) # 学習の状況を表示しない

    oof[valid_index] = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)



round(accuracy_score(Y_train, oof), 3)
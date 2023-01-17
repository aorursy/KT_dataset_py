!pip install optuna==2.0
import pandas as pd

import numpy as np

import lightgbm as lgb

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

import optuna
optuna.__version__
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

sample_submission = pd.read_csv('../input/titanic/gender_submission.csv')
# SexとEmbarkedのOne-Hotエンコーディング

train = pd.get_dummies(train, columns=['Sex', 'Embarked'])

test = pd.get_dummies(test, columns=['Sex', 'Embarked'])



# 不要な列の削除

train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)



# trainの表示

display(train.head())



X_train = train.drop(['Survived'], axis=1)  # X_trainはtrainのSurvived列以外

y_train = train['Survived']  # Y_trainはtrainのSurvived列
def objective(trial):

    kf = KFold(n_splits=3)

    gbm = lgb.LGBMClassifier(objective='binary')

    oof = np.zeros(len(train))



    for fold, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):

        train_x, valid_x = X_train.iloc[train_index], X_train.iloc[valid_index]

        train_y, valid_y = y_train[train_index], y_train[valid_index]

        gbm = lgb.LGBMClassifier(objective='binary',

                                 reg_alpha=trial.suggest_loguniform('reg_alpha', 1e-4, 100.0),

                                 reg_lambda=trial.suggest_loguniform('reg_lambda', 1e-4, 100.0),

                                 num_leaves=trial.suggest_int('num_leaves', 10, 40),

                                 silent=True)

        gbm.fit(train_x, train_y, eval_set = [(valid_x, valid_y)],

                early_stopping_rounds=20,

                verbose=-1) # 学習の状況を表示しない

        oof[valid_index] = gbm.predict(valid_x)



    accuracy = accuracy_score(y_train, oof)

    return 1.0 - accuracy
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

# シードを固定。

# 参考：https://qiita.com/phorizon20/items/1b795beb202c2dc378ed



study.optimize(objective, n_trials=50)
optuna.importance.get_param_importances(study)
fig = optuna.visualization.plot_param_importances(study)

fig.show()
importances = optuna.importance.get_param_importances(study)
from collections import OrderedDict



importances = OrderedDict(reversed(list(importances.items())))

importance_values = list(importances.values())

param_names = list(importances.keys())
importances, importance_values, param_names
import matplotlib.pyplot as plt

plt.barh(range(len(importance_values)), importance_values)

plt.yticks(range(3), param_names)

plt.title('Hyperparameter importance');
def objective(trial):

    kf = KFold(n_splits=3)

    oof = np.zeros(len(train))



    for fold, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):

        train_x, valid_x = X_train.iloc[train_index], X_train.iloc[valid_index]

        train_y, valid_y = y_train[train_index], y_train[valid_index]

        dtrain = lgb.Dataset(train_x, label=train_y)

        dvalid = lgb.Dataset(valid_x, label=valid_y)

        param = {

            'objective': 'binary',

            'metric': 'binary_logloss',

            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),

            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),

            'num_leaves': trial.suggest_int('num_leaves', 2, 256),

            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),

            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),

            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

        }

        

        gbm = lgb.train(param, dtrain, valid_sets=[dvalid], num_boost_round=100, early_stopping_rounds=20, verbose_eval=-1)

        oof[valid_index] = gbm.predict(valid_x)

    

    accuracy = accuracy_score(y_train, np.rint(oof))

    return accuracy

     

study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=10)

 

print('Number of finished trials:', len(study.trials))

print('Best trial:', study.best_trial.params)
fig = optuna.visualization.plot_param_importances(study)

fig.show()
study = optuna.create_study()
import optuna.integration.lightgbm as lgb



def main():

    kf = KFold(n_splits=3)

    oof = np.zeros(len(train))

    lgb_train = lgb.Dataset(X_train, y_train)

    param = {

        'objective': 'binary',

        'metric': 'binary_logloss',

        'verbosity': -1

    }

    

    tuner_cv = lgb.LightGBMTunerCV(

        param,

        lgb_train,

        num_boost_round=100,

        early_stopping_rounds=20,

        verbose_eval=20,

        folds=kf,

        study=study

    )

    

    tuner_cv.run()



    print(f'Best score: {tuner_cv.best_score}')

    print('Best params:')

    print(tuner_cv.best_params)



if __name__ == '__main__':

    main()
fig = optuna.visualization.plot_param_importances(study)

fig.show()



# lgb.LightGBMTunerCV では、importanceは表示できないんでしょうか？
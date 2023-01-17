# 特徴量の準備



import numpy as np

import pandas as pd



train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")



data = pd.concat([train, test], sort=False)



data['Sex'].replace(['male','female'], [0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(data['Age'].median(), inplace=True)

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
data.head()
delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True)



train = data[:len(train)]

test = data[len(train):]



y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)
X_train.head()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)
# カテゴリ変数の指定

categorical_features = ['Embarked', 'Pclass', 'Sex']
params = {

    'objective': 'binary'

}
import lightgbm as lgb





lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)



model = lgb.train(

    params, lgb_train,

    valid_sets=[lgb_train, lgb_eval],

    verbose_eval=10,

    num_boost_round=1000,

    early_stopping_rounds=10

)



y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred[:10]
params = {

    'objective': 'binary',

    'max_bin': 300,

    'learning_rate': 0.05,

    'num_leaves': 40

}
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)



model = lgb.train(

    params, lgb_train,

    valid_sets=[lgb_train, lgb_eval],

    verbose_eval=10,

    num_boost_round=1000,

    early_stopping_rounds=10

)



y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred[:10]
y_pred = (y_pred > 0.5).astype(int)

y_pred[:10]
sub = gender_submission



sub['Survived'] = y_pred

sub.to_csv("submission_lightgbm_handtuning.csv", index=False)



sub.head()
# KaggleのKernelに搭載済

import optuna

from sklearn.metrics import log_loss





def objective(trial):

    params = {

        'objective': 'binary',

        'max_bin': trial.suggest_int('max_bin', 255, 500),

        'learning_rate': 0.05,

        'num_leaves': trial.suggest_int('num_leaves', 32, 128),

    }

    

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)



    model = lgb.train(

        params, lgb_train,

        valid_sets=[lgb_train, lgb_eval],

        verbose_eval=10,

        num_boost_round=1000,

        early_stopping_rounds=10

    )



    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

    score = log_loss(y_valid, y_pred_valid)

    return score
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials=40)
study.best_params
params = {

    'objective': 'binary',

    'max_bin': study.best_params['max_bin'],

    'learning_rate': 0.05,

    'num_leaves': study.best_params['num_leaves']

}



lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)



model = lgb.train(

    params, lgb_train,

    valid_sets=[lgb_train, lgb_eval],

    verbose_eval=10,

    num_boost_round=1000,

    early_stopping_rounds=10

)



y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred > 0.5).astype(int)



sub['Survived'] = y_pred

sub.to_csv("submission_lightgbm_optuna.csv", index=False)



sub.head()
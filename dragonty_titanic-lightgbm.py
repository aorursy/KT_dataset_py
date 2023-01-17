import numpy as np

import pandas as pd



train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

y_train = train.Survived.values

train = train.drop("Survived", axis=1)

data = pd.concat([train, test], sort=False)

data.head()
def sep():

    print("========================================")

print(data.isnull().sum())

sep()

print(data.info())

sep()

print(data.describe())

sep()

print(data.columns)

sep()

print(data.nunique())

sep()

#print(data.Embarked.unique())

print(data.Embarked.value_counts())
data.Age.fillna(data.Age.median(), inplace=True)

data.Fare.fillna(data.Fare.mean(), inplace=True)

data.Embarked.fillna(data.Embarked.mode()[0], inplace=True)

data.Embarked = data.Embarked.map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data["FamilySize"] = data.SibSp + data.Parch + 1

data["IsAlone"] = 0

data.loc[data.FamilySize == 1, "IsAlone"] = 1

data.Sex.replace(["male", "female"], [0, 1], inplace=True)
print(data.isnull().sum())

sep()

data.head(10)
data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

data.head()
X_train = data[:len(train)]

X_test = data[len(train):]
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)

categorical_features = ["Embarked", "Pclass", "Sex"]
import optuna

from sklearn.metrics import log_loss





def objective(trial):

    params = {

        'objective': 'binary',

        'max_bin': trial.suggest_int('max_bin', 255, 500),

        'learning_rate': 0.05,

        'num_leaves': trial.suggest_int('num_leaves', 32, 128),

    }



    lgb_train = lgb.Dataset(X_train, y_train,

                            categorical_feature=categorical_features)

    lgb_eval = lgb.Dataset(X_valid, y_valid,

                           reference=lgb_train,

                           categorical_feature=categorical_features)



    model = lgb.train(params, lgb_train,

                      valid_sets=[lgb_train, lgb_eval],

                      verbose_eval=10,

                      num_boost_round=1000,

                      early_stopping_rounds=10)



    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

    score = log_loss(y_valid, y_pred_valid)

    return score
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials=40)

study.best_params
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)

params = {

    'objective': 'binary',

    'max_bin': 427,

    'learning_rate': 0.05,

    'num_leaves': 79

}

model = lgb.train(params, lgb_train,

                  valid_sets=[lgb_train, lgb_eval],

                  verbose_eval=10,

                  num_boost_round=1000,

                  early_stopping_rounds=5)



y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred > 0.5).astype(int)

y_pred
sub = gender_submission

sub["Survived"] = y_pred

sub.to_csv("submission.csv", index=False)
import pandas as pd
train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
sample=pd.read_csv('../input/titanic/gender_submission.csv')
train.Age.fillna(train.Age.mean(), inplace=True)
test.Age.fillna(test.Age.mean(), inplace=True)

train['Embarked']=train['Embarked'].map({'S':0,'C':1,'Q':2})
train['Sex']=train['Sex'].map({'male':0,'female':1})

test['Embarked']=test['Embarked'].map({'S':0,'C':1,'Q':2})
test['Sex']=test['Sex'].map({'male':0,'female':1})
X_train=train[['Pclass','Age','SibSp','Parch','Fare','Sex','Embarked']]
y_train=train['Survived']

X_test=test[['Pclass','Age','SibSp','Parch','Fare','Sex','Embarked']]
from sklearn.model_selection import train_test_split 

X_train,X_valid,y_train,y_valid=train_test_split(X_train,y_train,test_size=0.2,random_state=0)
import lightgbm as lgb

trains = lgb.Dataset(X_train, y_train)
valids = lgb.Dataset(X_valid, y_valid)
tests = lgb.Dataset(test)

import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import optuna


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):


    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, trains)
    preds = gbm.predict(X_valid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_valid, pred_labels)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
import optuna.integration.lightgbm as lgb

params = {
        'task': 'train',
        'objective': 'binary',    # 2値分類の指定
        'metric': 'binary_error', # 誤答率の割合
        'verbose': 1,
        'lambda_l1': 0.0003378640593022582,
        'lambda_l2': 0.000734325193694063,
        'num_leaves': 206,
        'feature_fraction': 0.9923599590563134,
        'bagging_fraction': 0.9383784759637969,
        'bagging_freq': 2,
        'min_child_samples': 24
}

model = lgb.train(params,
                  trains, 
                  valid_sets=valids,
                  verbose_eval=False,
                  num_boost_round=100,
                  early_stopping_rounds=5)
predict = model.predict(X_test, num_iteration=model.best_iteration)
PassengerId = test.PassengerId.values

submissions = pd.DataFrame({'PassengerId' : PassengerId,
                            'Survived' : predict})
submissions.Survived = submissions.Survived.apply(lambda x : 1 if x > 0.49 else 0)
submissions.to_csv('./v38_submissions.csv', index=False)
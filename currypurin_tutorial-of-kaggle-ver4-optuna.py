import pandas as pd

import numpy as np

import lightgbm as lgb

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

import optuna
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
kf = KFold(n_splits=3, shuffle=True, random_state=0)



# スコアとモデルを格納するリスト

score_list = []

test_pred = np.zeros((len(test), 3))



for fold_, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):

    train_x = X_train.iloc[train_index]

    valid_x = X_train.iloc[valid_index]

    train_y = y_train[train_index]

    valid_y = y_train[valid_index]

    

    print(f'fold{fold_ + 1} start')



    gbm = lgb.LGBMClassifier(objective='binary',

                             num_leaves=38,

                             reg_alpha=0.009253849015686676,

                             reg_lambda=3.47273669766293)  # パラメータを指定

    gbm.fit(train_x, train_y,

            eval_set = [(train_x, train_y), (valid_x, valid_y)],

            early_stopping_rounds=20,

            verbose= -1)

    

    oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)

    score_list.append(round(accuracy_score(valid_y, oof)*100,2))

    test_pred[:, fold_] = gbm.predict_proba(test)[:, 1]

    print(f'fold{fold_ + 1} end\n' )

print(score_list, '平均score', np.mean(score_list))

pred = (np.mean(test_pred, axis=1) > 0.5).astype(int)

sample_submission['Survived'] = pred

sample_submission.to_csv('optuna.csv', index=False)
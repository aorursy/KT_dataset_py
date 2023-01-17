import pandas as pd

import numpy as np

import lightgbm as lgb

from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings('ignore')

import os

os.listdir('../input')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/gender_submission.csv')



# Sexの変換

genders = {'female': 0, 'male':1}

train['Sex'] = train['Sex'].map(genders)

test['Sex'] = test['Sex'].map(genders)



# Embarkedの変換 今回はonehot encodingしない

embarked = {'S':0, 'C':1, 'Q':2}

train['Embarked'] = train['Embarked'].map(embarked)

test['Embarked'] = test['Embarked'].map(embarked)



# 不要な列の削除

train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

X_train = train.drop('Survived', axis=1)

y_train = train['Survived']
X_train.head()
# トレーニングデータをtrainとvalidに分割

train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.33, random_state=0)



# lab.Datasetを使って、trainとvalidを作っておく

lgb_train= lgb.Dataset(train_x, train_y, categorical_feature=['Sex', 'Embarked'])

lgb_valid = lgb.Dataset(valid_x, valid_y, categorical_feature=['Sex', 'Embarked'])



lgbm_params = {'objective': 'binary'}



# lgb.trainで学習

gbm = lgb.train(params=lgbm_params,

                train_set=lgb_train,

                valid_sets=[lgb_train, lgb_valid],

                early_stopping_rounds=20,

                verbose_eval=-1)



oof = gbm.predict(valid_x)



preds = (oof > 0.5).astype(int)

print('score', round(accuracy_score(valid_y, preds)*100,2))
# valid_xについて推論

oof = gbm.predict(valid_x)

print('score', round(accuracy_score(valid_y, (oof > 0.5).astype(int))*100,2))  # validのscore

test_pred = (gbm.predict(test) > 0.5).astype(int)

sample_submission['Survived'] = test_pred

sample_submission.to_csv('train_test_split.csv', index=False)  # score:75.119

gbm.feature_importance()

# importanceはtraining dataの列順に表示される
# 見やすくする

pd.DataFrame({'特徴': X_train.columns,

    'importance':gbm.feature_importance()}).sort_values('importance',

    ascending=False)
pd.DataFrame({'特徴': X_train.columns,

    'importance':gbm.feature_importance(importance_type='gain')}).sort_values('importance',

    ascending=False)
# トレーニングデータをtrainとvalidに分割

train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.33, random_state=0)



# LightGBMの分類器をインスタンス化

gbm = lgb.LGBMClassifier(objective='binary')  # , importance_type='gain'



# trainとvalidを指定し学習

gbm.fit(train_x, train_y,

        eval_set = [(train_x, train_y), (valid_x, valid_y)],

        categorical_feature=['Sex', 'Embarked'],

        early_stopping_rounds=20,

        verbose=-1)



# valid_xについて推論

oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)

print('score', round(accuracy_score(valid_y, oof)*100,2));  # validのscore
# 指定していないパラメータは、初期パラメータが表示される

gbm.get_params()
# GridSearchCVをimport

from sklearn.model_selection import GridSearchCV



gbm = lgb.LGBMClassifier(objective='binary')



# 試行するパラメータを羅列する

params = {

    'num_leaves': [20, 31, 40, 50],

    'reg_alpha': [0, 1, 10, 100],

    'reg_lambda': [0, 1, 10, 100],

}



grid_search = GridSearchCV(

                           gbm,  # 分類器を渡す

                           param_grid=params,  # 試行してほしいパラメータを渡す

                           cv=3,  # 3分割交差検証でスコアを確認

                          )



grid_search.fit(X_train, y_train)  # データを渡す



print(grid_search.best_score_)  # ベストスコアを表示

print(grid_search.best_params_)  # ベストスコアのパラメータを表示
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

                             num_leaves=20,

                             reg_alpha=0,

                             reg_lambda=10)  # パラメータを指定

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

sample_submission.to_csv('glid_search.csv', index=False)
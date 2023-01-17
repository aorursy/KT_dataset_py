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

Y_train = train['Survived']
X_train.head()
# トレーニングデータをtrainとvalidに分割

train_x, valid_x, train_y, valid_y = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)



# LightGBMの分類器をインスタンス化

gbm = lgb.LGBMClassifier(objective='binary')  # , importance_type='gain'

# trainとvalidを指定し学習

gbm.fit(train_x, train_y,

        eval_set = [(valid_x, valid_y)],

        categorical_feature=['Sex', 'Embarked'],  # カテゴリカル変数を指定する

        early_stopping_rounds=20,

        verbose=-1)



# valid_xについて推論

oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)

print('score', round(accuracy_score(valid_y, oof)*100,2));  # validのscore

# valid_xについて推論

oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)

print('score', round(accuracy_score(valid_y, oof)*100,2))  # validのscore

test_pred = gbm.predict(test, num_iteration=gbm.best_iteration_)

sample_submission['Survived'] = test_pred

sample_submission.to_csv('train_test_split.csv', index=False)  # score:75.119

# 指定しなかったパラメータを含め、パラメータを取得

gbm.get_params()
gbm.feature_importances_

# importanceはtraining dataの列順に表示される
# 見やすくする

pd.DataFrame({'特徴': X_train.columns,

    'importance':gbm.feature_importances_}).sort_values('importance',

    ascending=False)
gbm.feature_importances_# トレーニングデータをtrainとvalidに分割

train_x, valid_x, train_y, valid_y = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)



# LightGBMの分類器をインスタンス化

gbm = lgb.LGBMClassifier(objective='binary', importance_type='gain') 

# trainとvalidを指定し学習

gbm.fit(train_x, train_y,

        eval_set = [(valid_x, valid_y)],

        categorical_feature=['Sex', 'Embarked'],

        early_stopping_rounds=20,

        verbose=-1);
pd.DataFrame({'特徴': X_train.columns,

    'importance':gbm.feature_importances_}).sort_values('importance',

    ascending=False)
# トレーニングデータをtrainとvalidに分割

train_x, valid_x, train_y, valid_y = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)



# lab.Datasetを使って、trainとvalidを作っておく

lgb_train= lgb.Dataset(train_x, train_y, categorical_feature=['Sex', 'Embarked'])

lgb_eval = lgb.Dataset(valid_x, valid_y, categorical_feature=['Sex', 'Embarked'])



lgbm_params = {'objective': 'binary'}

evals_result = {}



# lgb.trainで学習

gbm = lgb.train(params = lgbm_params,  # パラメータは辞書で渡す

                        train_set = lgb_train,

                        valid_sets=[lgb_train, lgb_eval],

                        early_stopping_rounds=20,

                        verbose_eval=5,

                        evals_result=evals_result)



# predictは、0から1の少数での出力値のnumpy arrayでの出力となる

oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration)



preds = (oof > 0.5).astype(int)

print('score', round(accuracy_score(valid_y, preds)*100,2))

import matplotlib.pyplot as plt



plt.plot(evals_result['training']['binary_logloss'], label='train_loss')

plt.plot(evals_result['valid_1']['binary_logloss'], label='valid_loss')

plt.title('train_loss and valid_olss')

plt.legend();
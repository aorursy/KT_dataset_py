# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd

train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")


#以下前処理処理

data = pd.concat([train, test], sort=False)
# concatは連結

data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
# male femaleをそれぞれ0と1に変換。
data['Embarked'].fillna(('S'), inplace=True)
# Embarkedの欠損値をSに置き換え。
# fillnaやdropnaはデフォルトでOptionのinplaceはFalse。Trueにしないとデータフレームに反映されない。Falseの使用する事例は不明。（てか、使いどころないのでは。）
data['Embarked'] = data['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
# Embarkedは乗船した港。
# EmbarkedのS C Qをそれぞれ0, 1, 2に変換。最後に変換した値をIntとしてCast。
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
# Fareの欠損値を平均値に置き換え。
data['Age'].fillna(data['Age'].median(), inplace=True)
# Ageの欠損値を中央値で置き換え。
data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
# FamilySizeを新設。SibSpが兄弟、姉妹の数でParchが親または子の数
data['IsAlone'] = 0
# IsAloneを新設し、0でInit。
data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
# locはデータを参照し代入。
data.head()
delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

train=data[:len(train)]
test=data[len(train):]
# concatで連結していたTrainとTestを分けた。

y_train = train['Survived']
# 教師データ
X_train = train.drop('Survived', axis=1)
# トレーニングデータ。Axisが1なので行での参照。
X_test = test.drop('Survived', axis=1)
# テストデータの作成。
X_train.head()
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)
categorical_features = ['Embarked', 'Pclass', 'Sex']
params = {'objective': 'binary'} #問題の定義（多値分類、二値分類…）
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

# trainメソッドはModelを返すメソッド。

y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred[:10]
params = {
    'objective':'binary',
    'max_bin':300,
    'learning_rate':0.05,
    'num_leaves':40
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

    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration) #学習モデルにデータを通して予測データを作る。
    score = log_loss(y_valid, y_pred_valid) #目的関数をバリデーションしたもののloglossとしている。
    return score
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)) #create_studyメソッドに渡して最適化するセッションを作成。
#samplers.RandomSamplerでランダムサンプリングしている。
study.optimize(objective, n_trials=40)
study.best_params
import math



def objective(trial):
    # trialは探索したものを記録しておくオブジェクト。
    """最小化する目的関数"""
    # 線形探索する際のパラメータが取りうる範囲（引数：変数名、下限、上限）
    x = trial.suggest_uniform('x', -5, +15)
    # デフォルトで最小化かつ現在は最小化のみのサポートなので符号を反転する
    return - math.exp(-(x - 2) ** 2) + math.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1) #目的関数の値


def main():
    # 最適化のセッションを作る
    study = optuna.create_study()
    # 100 回試行する（optimizeに目的関数と試行回数を渡す。）
    study.optimize(objective, n_trials=10)
    # 最適化したパラメータを出力する
    print('params:', study.best_params)
main()
res = []
for _ in range(5):
    study = optuna.create_study()
    # 100 回試行する（optimizeに目的関数と試行回数を渡す。）
    study.optimize(objective, n_trials=100)
    res.append(study.best_params)
for p in res:
    print(p)
res = []
for _ in range(5):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    # 100 回試行する（optimizeに目的関数と試行回数を渡す。）
    study.optimize(objective, n_trials=100)
    res.append(study.best_params)
for p in res:
    print(p)

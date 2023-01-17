import numpy as np

import pandas as pd

import pandas_profiling
!pwd
!ls ../input/titanic/
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

ans = pd.read_csv("../input/titanic/gender_submission.csv")
train.profile_report()
test.head()
data = pd.concat([train, test], sort=False)
data.isnull().sum()
data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
data['Embarked'].fillna('S', inplace=True)

data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
age_avg = data['Age'].mean()

age_std = data['Age'].std()

data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True)
train = data[:len(train)]

test = data[len(train):]
y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)
from sklearn.model_selection import train_test_split





# stratify=y_trainを指定して分割したときのyの0,1の個数の比率に偏りがないようにする

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=0, stratify=y_train)
import lightgbm as lgb



# 本当はOptuna使って最適化するべき（https://tech.preferred.jp/ja/blog/hyperparameter-tuning-with-optuna-integration-lightgbm-tuner/）

params = {

    'objective': 'binary',

    'max_bin': 300,

    'learning_rate': 0.05,

    'num_leaves': 40

}



lgb_train = lgb.Dataset(X_train, y_train,)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)



model = lgb.train(params, lgb_train,

                               valid_sets=[lgb_train, lgb_eval],

                               verbose_eval=10,

                               num_boost_round=1000,

                               early_stopping_rounds=10)



y_pred = model.predict(X_test, num_iteration=model.best_iteration)
ans['Survived'] = list(map(int, y_pred))

ans.to_csv('submission.csv', index=False)
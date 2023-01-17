# 以下から

# https://www.kaggle.com/dlarionov/titanic-xgboost

# https://www.kaggle.com/aashita/xgboost-model-with-minimalistic-features

import pandas as pd

import numpy as np

import re



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

import sklearn.preprocessing as preprocessing 

import warnings

warnings.filterwarnings('ignore')

import datetime

today = datetime.datetime.today().strftime("%Y-%m-%d")
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
#トレーニングデータの中身確認

print(len(train))

train.head()
#テストデータの中身確認

print(len(test))

test.head()
# 男性の乗客と比較するとはるかに高い割合で女性の乗客がタイタニック号の難破船を生き残ったことがわかる。

sns.barplot(x='Sex', y='Survived', data=train);
# チケットクラスも生存率の予測に役立つ指標になることがわかる。

sns.barplot(x='Pclass', y='Survived', data=train);
#ターゲットデータを切り離しておく

target = train.Survived.astype('int')

#ターゲットデータを切り話したDFを作っておく

pre_train = train.drop('Survived', axis=1)

#最終アウトプットのためにPassengerIdを取得

PassengerId = test.PassengerId
#トレーニングデータとテストデータを合体させたDFを持って置く（こうするのが個人的に好き）

merge_df = pd.concat([pre_train, test])

#トレーニングデータとテストデータをリストに入れておく（こうするのが個人的に好き）

tables = [train, test]
# 数値データとそうでないデータとなっているカラムを見てみる

numerical_features   = merge_df.select_dtypes(include=["float","int","bool"]).columns.values

categorical_features = merge_df.select_dtypes(include=["object"]).columns.values

print(numerical_features)

print(categorical_features)
#list(set(tables[1]['Cabin']))
for feature in categorical_features:

    # 'Cabin','Embarked'のnull値は一番多く出た種類の値に変える

    # 'Cabin','Embarked','Sex'をラベルエンコーディングで数値に変換する

    if feature in ['Cabin','Sex','Embarked']:

        for table in tables:

            print(feature)

            table[feature].fillna(table[feature].value_counts().idxmax(), inplace = True)

            le = preprocessing.LabelEncoder()

            le.fit(table[feature])

            table[feature] = le.transform(table[feature])

        merge_df[feature].fillna(merge_df[feature].value_counts().idxmax(), inplace = True)

        le = preprocessing.LabelEncoder()

        le.fit(merge_df[feature])

        merge_df[feature] = le.transform(merge_df[feature])

for table in tables:

    table['Age'].fillna(merge_df['Age'].median(), inplace = True)

merge_df['Age'].fillna(merge_df['Age'].median(), inplace = True)
tables[0].head()
features = [

    'Sex',

    'Pclass',

    'Age',

    'Cabin',

    'Embarked',

    'Parch'

]
train = tables[0]

X_train = train[features]

for f in features:

    X_train[f] = X_train[f].astype('int')

dtrain = xgb.DMatrix(data=X_train, label=target)

params = {

    "Objective": 'gbtree',

    "eval_metric": 'error',

    "eta": 0.1

}

# https://xgboost.readthedocs.io/en/latest/python/python_api.html

cv = xgb.cv(params=params, dtrain=dtrain, num_boost_round=300, nfold=5, seed=41, early_stopping_rounds= 10)

#cv.tail(1)
cv
# 今回の場合gradient boosted treesは11個で十分だそうなので11に。

xgbcl = XGBClassifier(n_estimators=11, seed=41)

xgbcl.fit(X_train, target)

print(xgbcl.score(X_train, target))
#xgb.to_graphviz(xgbcl)
test = tables[1]

X_test = test[features]

predictions = xgbcl.predict(X_test)

Predictions = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})

Predictions.to_csv(f'titanic_{today}_submission.csv', index=False)
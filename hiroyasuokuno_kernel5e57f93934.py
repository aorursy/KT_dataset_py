# データ処理のためのライブラリ

import pandas as pd

import numpy as np

# 可視化用のライブラリ

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()

sns.set_style(style='dark')

# 1セルの実行結果を複数表示するための便利設定

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%load_ext autoreload

%autoreload 2

df=pd.read_csv("../input/titanic/train.csv")

df.Embarked=df.Embarked.fillna("S")

#"S"が最も多かったので、欠損値に"S"を代入
 # 列Ageの欠損値をAgeの最頻値で穴埋め

df.Age=df.Age.fillna(24.0)
#整数化

df['Fare'] = df['Fare'].astype(int)

df['Age'] = df['Age'].astype(int)
#NaN とそれ以外の値の特徴量を作成する 1=NaN 0=それ以外



df['Cabin_nan'] = np.where(df['Cabin'].isna(), 1, 0)
#運賃をビニング 10分割にして処置

pd.cut(df['Fare'], 10, precision=0).value_counts(sort=False, dropna=False)
df['Fare_10'] = pd.cut(df['Fare'], 10, labels=False)
#年齢をビニング 10分割にして処置

pd.cut(df['Age'], 10, precision=0).value_counts(sort=False, dropna=False)
df['Age_10'] = pd.cut(df['Age'], 10, labels=False)
#ダミー変数化

df = pd.get_dummies(data=df, columns=['Sex'])
#ダミー変数化

df = pd.get_dummies(data=df, columns=['Embarked'])
df=df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin','Age','Fare'],axis=1)

# 上の6つのcolumn名をリストに。
df.head()
#データの分割を行う 訓練用データ 0.8 評価用データ 0.2



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

#説明変数セット

X = df[['Pclass', 'SibSp','Parch','Cabin_nan','Age_10','Fare_10','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']]

#目的変数セット

y = df['Survived']

#訓練データから擬似訓練データと擬似テストデータに分割する

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#ロジスティック回帰モデルのインスタンスを生成

lr = LogisticRegression()

#ロジスティック回帰モデルに擬似訓練データで学習させる

lr.fit(X_train, y_train)

#評価の実行（確率）

df1 = pd.DataFrame(lr.predict_proba(X_test))
df1
#ランダムフォレスト

from sklearn.metrics import roc_curve, auc, accuracy_score

from sklearn.ensemble import RandomForestRegressor 

clf = RandomForestRegressor(random_state=0,max_depth= 3, n_estimators = 93, max_features= 'auto')

clf = clf.fit(X_train, y_train)

pred = clf.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=1)

auc(fpr, tpr)

accuracy_score(y_test, pred.round(), normalize=False)
df_test = pd.read_csv("../input/titanic/test.csv")
df_test.Embarked=df_test.Embarked.fillna("S")



df_test.Age=df_test.Age.fillna(24.0)
df_test['Cabin_nan'] = np.where(df_test['Cabin'].isna(), 1, 0)

pd.cut(df_test['Fare'], 10, precision=0).value_counts(sort=False, dropna=False)
df_test['Fare_10'] = pd.cut(df_test['Fare'], 10, labels=False)

pd.cut(df_test['Age'], 10, precision=0).value_counts(sort=False, dropna=False)

df_test['Age_10'] = pd.cut(df_test['Age'], 10, labels=False)
df_test = pd.get_dummies(data=df_test, columns=['Sex'])

df_test = pd.get_dummies(data=df_test, columns=['Embarked'])

df_test=df_test.drop(columns=[ 'Name', 'Ticket', 'Cabin','Age','Fare'],axis=1)
df_test.head()
#  提出する

submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": accuracy_score

    })

submission.to_csv("submission.csv", index=False)
submission = pd.read_csv("../input/titanic/test.csv", index_col="PassengerId")

print(submission.shape)

submission.head()
# kaggleに提出するcsv作成

submission.to_csv("submit_181211.csv")
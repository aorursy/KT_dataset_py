# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv') #訓練用データ

test_data = pd.read_csv('../input/test.csv') #テスト用データ
print(train_data.columns)
train_data.head()
print(test_data.columns)
test_data.head()
train_data.isnull().any()
test_data.isnull().any()
age_mean = pd.concat([train_data['Age'],test_data['Age']]).mean() #訓練用データとテスト用データの'Age'を結合して平均を取る

train_data.fillna({'Age':age_mean},inplace=True)

train_data.isnull().any() #訓練用データの'Age'に欠損値が無くなったことを確認
test_data.fillna({'Age':age_mean}, inplace=True)

test_data.isnull().any() #テスト用データの'Age'に欠損値が無くなったことを確認
fare_mean = pd.concat([train_data['Fare'],test_data['Fare']]).mean()

test_data.fillna({'Fare':fare_mean},inplace=True) #Fareが欠落していたのはテスト用データだけ

test_data.isnull().any()
pd.concat([train_data['Cabin'],test_data['Cabin']]).value_counts()
most_existing_cabin = pd.concat([train_data['Cabin'],test_data['Cabin']]).value_counts().index[0]

train_data.fillna({'Cabin':most_existing_cabin},inplace=True) 

train_data.isnull().any()
test_data.fillna({'Cabin':most_existing_cabin},inplace=True) 

test_data.isnull().any()
pd.concat([train_data['Embarked'],test_data['Embarked']]).value_counts()
most_existing_embarked = pd.concat([train_data['Embarked'],test_data['Embarked']]).value_counts().index[0]
train_data.fillna({'Embarked':most_existing_embarked},inplace=True) #Embarkedが欠損しているのは訓練用データのみ

train_data.isnull().any()
train_data.isnull().any()
test_data.isnull().any()
train_X = train_data[['Pclass','Sex','Age','SibSp','Parch','Fare']] #全訓練データの中から対象のカラムのみ抜き出す

train_X.head()
train_y = train_data[['Survived']] #正解ラベルである'Survived'も切り出しておく

train_y.head()
test_X = test_data[['Pclass','Sex','Age','SibSp','Parch','Fare']] #全テストデータの中から対象のカラムのみ抜き出す

test_X.head()
pd.concat([train_X['Sex'],test_X['Sex']]).value_counts() #'Sex'のデータの中身を確認
pd.get_dummies(train_X['Sex']).head() #カラム'Sex'のダミー変数を取得して先頭を表示
pd.get_dummies(train_X['Sex'],drop_first=True).head() #drop_first=Trueを指定することにより先頭のカラムを落とす
train_X = train_X.join(pd.get_dummies(train_X['Sex'],drop_first=True)) #まずダミー変数を結合する

train_X.head()
train_X.drop(['Sex'], axis=1, inplace=True) #元のカラムである'Sex'は削除する

train_X.head()
# テストデータについても同じことを行う（ワンライナー）

test_X = test_X.join(pd.get_dummies(test_X['Sex'],drop_first=True)).drop(['Sex'],axis=1)

test_X.head()
import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
sns.heatmap(train_y.join(train_X).corr(),annot=True,cmap='RdYlGn',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
g = sns.factorplot(x="male", y="Survived",  data=train_y.join(train_X),size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.factorplot(x="Pclass", y="Survived",  data=train_y.join(train_X),size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
from sklearn.preprocessing import StandardScaler #scikit-learnというライブラリから標準化用のクラスをインポートする



sc = StandardScaler() #標準化クラスのインスタンス化

train_X = sc.fit_transform(train_X) #訓練データを標準化して上書きする 

test_X = sc.fit_transform(test_X) #同じくテストデータを標準化して上書きする
from sklearn.model_selection import GridSearchCV #交差検証法を用いてハイパーパラメータを検証するライブラリ



from sklearn.svm import SVC # カーネルを限定しないSVMモデルのライブラリ



# ハイパーパラメータの候補

param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1 ,1],

                  'C': [1, 10, 100, 1000]}



# GridSearchCVライブラリのインスタンス化

modelsvm = GridSearchCV(SVC(random_state=0),param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 2, verbose = 1)



# すべてのパラメータの組み合わせで交差検証を行い最も優れたパラメータを探し出す（=Grid Search）

modelsvm.fit(train_X,train_y)



# 最もスコアが高かったパラメータの組み合わせ

print(modelsvm.best_estimator_)

# 最も高かったスコア

print(modelsvm.best_score_)
import xgboost as xgb #scikit-learnのXGBClassifierはxgboostのラッパーなので、まずはxgboostをインポートする

from xgboost.sklearn import XGBClassifier #scilit-learnのXGBoost分類機



#XGBoostのパラメータ候補

param_grid = {

    'learning_rate':[0.1],

    'n_estimators':[1000],

    'max_depth':[3,5],

    'min_child_weight':[1,2,3],

    'max_delta_step':[5],

    'gamma':[0,3,10],

    'subsample':[0.8],

    'colsample_bytree':[0.8],

    'objective':['binary:logistic'],

    'nthread':[4],

    'scale_pos_weight':[1],

    'num_round':[10000],

    'seed':[0]

}



# GridSearchCVライブラリのインスタンス化

# 使うモデルがSVCからXGBClassifierに変わっただけで、それ以外のインタフェースは変わらない

modelxgb = GridSearchCV(XGBClassifier(),param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= -1, verbose = 1)



# 以下、全てSVCの場合と同じ



# すべてのパラメータの組み合わせで交差検証を行い最も優れたパラメータを探し出す（=Grid Search）

modelxgb.fit(train_X,train_y)



# 最もスコアが高かったパラメータの組み合わせ

print(modelxgb.best_estimator_)

# 最も高かったスコア

print(modelxgb.best_score_)
test_y = modelxgb.best_estimator_.predict(test_X) #予測



#提出用データの作成

submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": test_y}) #予測した結果とPassengerIdを結合する



#表示

print(submission)



#CSVファイルに出力

submission.to_csv("titanic_submission.csv", index=False)
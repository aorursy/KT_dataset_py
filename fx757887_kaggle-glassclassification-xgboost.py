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
#ライブラリ

import pandas as pd

import numpy as np



#可視化ライブラリ

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

%matplotlib inline



#Scikit-learn

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score, GridSearchCV



#XGBoost

import xgboost as xgb

from xgboost import XGBClassifier

#CSVファイル読込

glass=pd.read_csv("/kaggle/input/glass/glass.csv")

#先頭5行

glass.head()
#index 0の　Na~Feまでの総和計算

print(glass.iloc[0][1:-1].sum())

#index 1の　Na~Feまでの総和計算

print(glass.iloc[1][1:-1].sum()) 
glass["Type"].value_counts()
#欠損値確認

glass.isnull().sum()
#要約統計量確認

glass.describe()
#ターゲットクラス分布確認

sns.factorplot(x="Type", kind="count", data=glass)

plt.title("Glass Type Counts")
#散布図行列をプロット

sns.pairplot(glass[["RI", "Na", "Mg", "Al", "Type"]], hue="Type", diag_kind="hist")
#散布図行列をプロット

sns.pairplot(glass[["Si","K","Ca","Ba","Fe","Type"]],hue="Type",diag_kind="hist")
features=glass.iloc[:, 0:4].columns



plt.figure(figsize=(20,9*5))

gs=gridspec.GridSpec(4,1)

for i, col in enumerate(glass[features]):

    plt.title("Glass features")

    ax=plt.subplot(gs[i])

    sns.boxplot(x=glass["Type"], y=glass[col], palette="Set2", linewidth=1.0)

    sns.swarmplot(x=glass["Type"], y=glass[col], color="0.5")
features=glass.iloc[:, 4:8].columns



plt.figure(figsize=(20,9*5))

gs=gridspec.GridSpec(4,1)

for i, col in enumerate(glass[features]):

    plt.title("Glass features")

    ax=plt.subplot(gs[i])

    sns.boxplot(x=glass["Type"], y=glass[col], palette="Set2", linewidth=1.0)

    sns.swarmplot(x=glass["Type"], y=glass[col], color="0.5")
#訓練・テストデータ分割

train_set, test_set=train_test_split(glass, test_size=0.2, random_state=42)
#訓練データの特徴量・ターゲット

X_train=train_set.drop("Type",axis=1)

y_train=train_set["Type"].copy()

#テストデータの特徴量・ターゲット

X_test=test_set.drop("Type",axis=1)

y_test=test_set["Type"].copy()
#その１-ロジスティック回帰

#ロジスティック回帰のモデル訓練

LR=LogisticRegression(random_state=42)

LR.fit(X_train,y_train)
#テストデータの推測と評価

LR_pred_test=LR.predict(X_test)
#混同行列確認

confusion_matrix(y_test,LR_pred_test)
#正解率表示

accuracy_score(y_test,LR_pred_test)
#その２-ランダムフォレスト

#ランダムフォレスト

RF=RandomForestClassifier(random_state=42)
#ハイパーパラメータ交差検証

param_grid=[{

    "n_estimators":[5,10,50,100],

    "min_samples_split":[2,5,10],

    "bootstrap":["Auto","sqrt"],

    "min_samples_leaf":[1,5,10],

    "max_depth":[10,50,90],

    "max_features":["auto","sqrt"],

    "random_state":[42]

}]
#グリッドサーチＣＶで交差検証

RF_CV=GridSearchCV(estimator=RF,param_grid=param_grid,cv=5)

RF_CV.fit(X_train,y_train)
#最適なハイパーパラメータ表示

print(RF_CV.best_params_)
#テストデータの推測と評価（ランダムフォレスト回帰）

RF_pred_test=RF_CV.predict(X_test)
#混同行列表示

confusion_matrix(y_test, RF_pred_test)
#正解率表示

accuracy_score(y_test, RF_pred_test)
from sklearn.svm import SVC
#サポートベクターマシン

SV=SVC(random_state=42)
#ハイパーパラメータの交差検証

param_grid=[{

    "C":[0.1, 1, 10],

    "gamma":[0.01, 0.1, 1],

    "kernel":["rbf", "poly", "linear", "sigmoid"],

    "random_state":[42]

}]
#グリッドサーチＣＶで交差検証

SV_CV=GridSearchCV(estimator=SV,param_grid=param_grid,cv=5)

SV_CV.fit(X_train,y_train)
#最適なハイパーパラメータ表示

print(SV_CV.best_params_)
#テストデータの推測と評価（ランダムフォレスト回帰）

SV_pred_test=SV_CV.predict(X_test)
#混同行列表示

confusion_matrix(y_test,SV_pred_test)
#正解率表示

accuracy_score(y_test, SV_pred_test)
#ナイーブベイズ

NB=GaussianNB()
#モデル訓練

NB.fit(X_train,y_train)
#テストデータの推測と評価（ランダムフォレスト回帰）

NB_pred_test=NB.predict(X_test)
#混同行列表示

confusion_matrix(y_test, NB_pred_test)
#正解率表示

accuracy_score(y_test, NB_pred_test)
#XGBoost

##XGB Boost Model

xgboost=XGBClassifier(random_state=42)
#ハイパーパラメータ交差検証

param_grid=[{

    "n_estimators":[100,300,500],

    "max_depth":[6,10],

    "min_child_weight":[1,10],

    "subsample":[0.9, 1.0],

    "colsample_bytree":[0.9, 1.0],

    "random_state":[42]

}]
#グリッドサーチＣＶで交差検証

xgboost_CV=GridSearchCV(estimator=xgboost, param_grid=param_grid, cv=5)

xgboost_CV.fit(X_train,y_train)
#最適なハイパーパラメータ表示

print(xgboost_CV.best_params_)
#テストデータの推測と評価(xgboost)

xgboost_pred_test=xgboost_CV.predict(X_test)
#混同行列表示

confusion_matrix(y_test, xgboost_pred_test)
#正解率表示

accuracy_score(y_test, xgboost_pred_test)
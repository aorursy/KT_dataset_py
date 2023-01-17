# csvの読み込みなどでデータを取り扱う.
import pandas as pd
# 多次元配列を扱う数値演算ライブラリ.
import numpy as np
# 正規表現.
import re
# csvの読み込みなどでデータを取り扱う.
import pandas as pd
# 多次元配列を扱う数値演算ライブラリ.
import numpy as np
# 正規表現.
import re
# 機械学習系ライブラリscikit-learn
import sklearn
# アンサンブル学習.
import xgboost as xgb
# チャートを描画.
import matplotlib.pyplot as plt
import plotly
plotly.offline.init_notebook_mode(connected=False)
# グラフをプロット.
import matplotlib.pyplot as plt
# Matplotlibをきれいにしたラッパー.
import seaborn as sns
import math
import warnings
warnings.filterwarnings('ignore')

# サイキット・ラーン.
# 機械学習ライブラリ.
# * RandomForestClassifier ランダムフォレスト.
# * AdaBoostClassifier エイダブースト.
# * GradientBoostingClassifier 勾配ブースト.
# * ExtraTreesClassifier 
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
# * サポートベクターマシン.
from sklearn.svm import SVC
# * クロスバリデーション.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

PassengerId = test["PassengerId"]
train.head()
train['Sex'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
train["Family"] = train["SibSp"] + train["Parch"] + 1
test["Family"] = test["SibSp"] + test["Parch"] + 1
train["Alone"] = 0
test["Alone"] = 0
train.loc[train["Family"] == 1, "Alone"] = 1
test.loc[test["Family"] == 1, "Alone"] = 1
train.isnull().sum()
test.isnull().sum()
train["hasCabin"] = [0 if type(x)==float else 1 for x in train["Cabin"]]
test["hasCabin"] = [0 if type(x)==float else 1 for x in test["Cabin"]]
# [半角スペース]で始まり,アルファベット数文字のあとに[.ピリオド]があればそのデータを作成.
train["title"] = ["" if re.search(' ([A-Za-z]+)\.', x)==False else re.search(' ([A-Za-z]+)\.', x).group(1) for x in train["Name"]]
test["title"] = ["" if re.search(' ([A-Za-z]+)\.', x)==False else re.search(' ([A-Za-z]+)\.', x).group(1) for x in test["Name"]]
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

train["title"] = train["title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train["title"] = train["title"].replace('Mlle', 'Miss')
train["title"] = train["title"].replace('Ms', 'Miss')
train["title"] = train["title"].replace('Mme', 'Mrs')
train["title"] = train["title"].map(title_mapping)
train["title"] = train["title"].fillna(0)
test["title"] = train["title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test["title"] = train["title"].replace('Mlle', 'Miss')
test["title"] = train["title"].replace('Ms', 'Miss')
test["title"] = train["title"].replace('Mme', 'Mrs')
test["title"] = train["title"].map(title_mapping)
test["title"] = train["title"].fillna(0)
sns.jointplot(x="Age",y="SibSp",data=train)
sns.jointplot(x="Age",y="Parch",data=train)
age_cate = []
for index,x in train.iterrows():
    ap = 0
    if x["Age"] > 0:
        # 年齢が入力されている場合.
        ap = np.floor(x["Age"] / 10.0).astype(int)
    else:
        # 年齢が入力されていない場合.
        ap = np.floor( 1 / (x["SibSp"] + 1) + 1 / (x["Parch"] + 1)).astype(int)
        # titleを持っている.
        if x["title"] != 0:
            ap += 1
        # Aloneである.
        if x["Alone"] != 0:
            ap += 1
    age_cate.append(ap)
train["AgeCategory"] = age_cate

age_cate_test = []
for index,x in test.iterrows():
    ap = 0
    if x["Age"] > 0:
        # 年齢が入力されている場合.
        ap = np.floor(x["Age"] / 10.0).astype(int)
    else:
        # 年齢が入力されていない場合.
        ap = np.floor( 1 / (x["SibSp"] + 1) + 1 / (x["Parch"] + 1)).astype(int)
        # titleを持っている.
        if x["title"] != 0:
            ap += 1
        # Aloneである.
        if x["Alone"] != 0:
            ap += 1
    age_cate_test.append(ap)
test["AgeCategory"] = age_cate_test
train["Embarked"] = train["Embarked"].fillna('S')
train["Embarked"] = train["Embarked"].map( {'S': 1, 'C': 2, 'Q': 3} ).astype(int)
test["Embarked"] = test["Embarked"].fillna('S')
test["Embarked"] = test["Embarked"].map( {'S': 1, 'C': 2, 'Q': 3} ).astype(int)
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

train["FareCategory"] = np.floor(train["Fare"] / 20.0).astype(int)
test["FareCategory"] = np.floor(test["Fare"] / 20.0).astype(int)
drop_columns = ["PassengerId", "Ticket", "SibSp", "Parch", "Cabin", "Age", "Name", "Fare"]
train = train.drop(drop_columns, axis=1)
test = test.drop(drop_columns, axis=1)
train.head()
plt.figure(figsize=(14,12))
sns.heatmap(
    data = train.astype(float).corr(),
    vmax = 1.0, 
    vmin = -1.0, 
    square = True,
    cmap = plt.get_cmap("Spectral_r"),
    annot = True
)

train_suv = train[["Survived"]]
a_train = train.drop(["Survived"], axis = 1);
x_train, x_test, y_train, y_test = train_test_split(
    a_train,
    train_suv,
    test_size=0.1,
    shuffle = False,
    random_state=0
)
print("count x_train:{} & x_test:{}".format(len(x_train), len(x_test)))
RANDOM_SEED = 1
rf_p = {
    "random_state":RANDOM_SEED,
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
    'max_depth': 20,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

rf = RandomForestClassifier(**rf_p)
rf_i = rf.fit(x_train, y_train).feature_importances_
et_p = {
    'n_jobs': -1,
    'n_estimators':500,
    "random_state":RANDOM_SEED,
    'max_depth': 20,
    'min_samples_leaf': 2,
    'verbose': 0
}
et = ExtraTreesClassifier(**et_p)
et_i = et.fit(x_train, y_train).feature_importances_
ada_p = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}
ada = AdaBoostClassifier(**ada_p)
ada_i = ada.fit(x_train, y_train).feature_importances_
gb_p = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 20,
    'min_samples_leaf': 2,
    'verbose': 0
}
gb = GradientBoostingClassifier(**gb_p)
gb_i = gb.fit(x_train, y_train).feature_importances_
features_dataframe = pd.DataFrame({
    'features':x_train.columns,
    'RandomForest':rf_i,
    'ExtraTree':et_i,
    'AdaBoost':ada_i,
    'GradientBoost':gb_i
});
features_dataframe['mean'] = features_dataframe.mean(axis=1)
features_dataframe.head()

fig = plt.figure(figsize=(12,5))
plt.bar(features_dataframe['features'].values, features_dataframe['mean'].values)
plt.xticks(rotation=45)
#scoring = "f1_macro"
scoring = "f1_weighted"

cv_count = 10
rf_score = cross_val_score(rf, x_train, y_train, cv=cv_count, scoring=scoring)
mean = rf_score.mean()
med = np.median(rf_score)
data = [
    plotly.graph_objs.Bar(x=np.arange(cv_count), y=rf_score, name="試行"),
    plotly.graph_objs.Scatter(x=np.arange(cv_count),y=np.full(cv_count, mean), name="平均"),
    plotly.graph_objs.Scatter(x=np.arange(cv_count),y=np.full(cv_count, med), name="中央値")
]
layout = plotly.graph_objs.Layout(
    title="CV Score - RandomForest Count{}".format(cv_count),
    xaxis={"title":"回数"},
    yaxis={"title":"スコア", "range":[0.7, 1]},
)
fig = plotly.graph_objs.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)
et_score = cross_val_score(et, x_train, y_train, cv=cv_count, scoring=scoring)
mean = et_score.mean()
med = np.median(et_score)
data = [
    plotly.graph_objs.Bar(x=np.arange(cv_count), y=et_score, name="試行"),
    plotly.graph_objs.Scatter(x=np.arange(cv_count),y=np.full(cv_count, mean), name="平均"),
    plotly.graph_objs.Scatter(x=np.arange(cv_count),y=np.full(cv_count, med), name="中央値")
]
layout = plotly.graph_objs.Layout(
    title="CV Score - Extra Trees Count{}".format(cv_count),
    xaxis={"title":"回数"},
    yaxis={"title":"スコア", "range":[0.7, 1]},
)
fig = plotly.graph_objs.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)
ada_score = cross_val_score(ada, x_train, y_train, cv=cv_count, scoring=scoring)
mean = ada_score.mean()
med = np.median(ada_score)
data = [
    plotly.graph_objs.Bar(x=np.arange(cv_count), y=ada_score, name="試行"),
    plotly.graph_objs.Scatter(x=np.arange(cv_count),y=np.full(cv_count, mean), name="平均"),
    plotly.graph_objs.Scatter(x=np.arange(cv_count),y=np.full(cv_count, med), name="中央値")
]
layout = plotly.graph_objs.Layout(
    title="CV Score - Ada Boost Count{}".format(cv_count),
    xaxis={"title":"回数"},
    yaxis={"title":"スコア", "range":[0.7, 1]},
)
fig = plotly.graph_objs.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)
gb_score = cross_val_score(gb, x_train, y_train, cv=cv_count, scoring=scoring)
mean = gb_score.mean()
med = np.median(gb_score)
data = [
    plotly.graph_objs.Bar(x=np.arange(cv_count), y=gb_score, name="試行"),
    plotly.graph_objs.Scatter(x=np.arange(cv_count),y=np.full(cv_count, mean), name="平均"),
    plotly.graph_objs.Scatter(x=np.arange(cv_count),y=np.full(cv_count, med), name="中央値")
]
layout = plotly.graph_objs.Layout(
    title="CV Score - Gradient Boost Count{}".format(cv_count),
    xaxis={"title":"回数"},
    yaxis={"title":"スコア", "range":[0.7, 1]},
)
fig = plotly.graph_objs.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)
prediction = et.predict(test)
Submission = pd.DataFrame(
    {
        'PassengerId': PassengerId,
        'Survived': prediction
    }
)
Submission.to_csv("Submission.csv", index=False)
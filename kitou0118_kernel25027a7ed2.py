import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.preprocessing as sp

import os

import re

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import sklearn

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler

import xgboost as xgb



train = pd.read_csv('../input/train.csv',index_col=0)

test = pd.read_csv('../input/test.csv',index_col=0)
#train.isnull().sum()

#test.isnull().sum()

#train.describe()

train_index = train.index

test_index = test.index
train = pd.concat([train,test],sort=False)

marge_index = train.index
#Nameから名称(Mr,Msなど)を抽出 Name:Braund, Mr. Owen Harris

#,で分割  Braund, Mr. Owen Harris ⇒ Braund  /  Mr. Owen Harris

df_work = train["Name"].str.split(",",expand=True)

#.で分割   Mr. Owen Harris　⇒ Mr / Owen Harris

df_work = df_work[1].str.split(".",expand=True)

#もとのDFに結合

train = pd.concat([train,df_work[0]],axis=1)

#結合した列名をtitleに変更

train.rename(columns={0:"title"},inplace=True)

#名称ごとに年齢の平均を取得（名称のリストを作るために一旦titleはindexにしない

df_title = train.groupby(["title"],as_index=False).mean()

#名称のリストを作成

title_list = df_title["title"].values.tolist()

#titleをindexにする

df_title = df_title.set_index("title")

# titleごとに取得した年齢の平均を Age に代入（nullのみ）

df_age = pd.DataFrame()

for title in title_list:

 train_nan = train.loc[(train["title"] == title)]

 train_nan = train_nan.fillna({"Age": df_title.loc[title,"Age"]})

 df_age = df_age.append(train_nan)

train = train.drop(["Age"],axis=1).join(df_age["Age"])
train["Cabin"] = train["Cabin"].fillna(0)

train["Cabin_flg"] = train["Cabin"].where(train["Cabin"] == 0,1)
train["Fare"] = train["Fare"].fillna(train['Fare'].median())

#train["Embarked"].value_counts() Embarkedで最も出頻度が高い"S"でnanを埋める

train["Fare"] = train["Fare"].fillna("S")

train["Age_bin"] = pd.cut(train["Age"],10)

train["Fare_bin"] = pd.cut(train["Fare"],50)

train["Family"] = train["Parch"] + train["SibSp"]

train.drop(["Name","Cabin","Ticket"],axis=1,inplace=True)

train["Age_Fare_mul"] = train["Age"] * train["Fare"]
#多項式特徴量生成

#引数に対して、指定した次数までの多項値を作成する。例えば、引数にx,yを渡し、次数に2を指定した場合は以下のようになる。

#[x, y] ⇒ [1, x, y, x*2, x*y, y*y]

#pf = sp.PolynomialFeatures(degree=4,include_bias=False)

#pf.fit(train[["Age"]])

#train = pd.concat([train,pd.DataFrame(pf.transform(train[["Age"]]))],axis=1)
# 女性、客室クラスが上のほうが生存率が高い

#sns.barplot(x='Sex', y='Survived', hue='Pclass', data=train)
#sns.barplot(x='SibSp', y='Survived',data=train)
#sns.barplot(x='Parch', y='Survived',data=train)
train= pd.get_dummies(train,columns=["Sex","Embarked","Age_bin","Fare_bin","title"])
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train[['Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Age', 'Family','Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S','title_ Capt', 'title_ Col', 'title_ Don', 'title_ Dona', 'title_ Dr',

       'title_ Jonkheer', 'title_ Lady', 'title_ Major', 'title_ Master',

       'title_ Miss', 'title_ Mlle', 'title_ Mme', 'title_ Mr', 'title_ Mrs',

       'title_ Ms', 'title_ Rev', 'title_ Sir', 'title_ the Countess','Age_Fare_mul','Cabin_flg']].astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
col_list = train.columns.values

for col_temp in col_list:

 col_rename = re.sub(r"\[|\]|<","",col_temp)

 train.rename(columns={col_temp:col_rename},inplace=True)
temp_df = train.loc[:,train.columns.str.startswith("Age_")]

temp_list = temp_df.columns.values

train_nbin = train.drop(temp_list,axis=1)
temp_df = train.loc[:,train.columns.str.startswith("Fare_")]

temp_list = temp_df.columns.values

train_nbin = train_nbin.drop(temp_list,axis=1)
#データの正規化 0～1の間に納める

train = train.astype("float64")

ms = sp.MinMaxScaler()

ms.fit(train)

train_min = pd.DataFrame(ms.transform(train),index=marge_index)

col_list = train.columns.values

train_min.columns = col_list
#データの正規化 0～1の間に納める

train = train.astype("float64")

scaler = StandardScaler()

scaler.fit(train)

train_scale = pd.DataFrame(scaler.transform(train),index=marge_index)

col_list = train.columns.values

train_scale.columns = col_list

train_scale["Survived"] = train["Survived"]
#データの正規化 0～1の間に納める

col_list = train_nbin.columns.values

train_nbin_min = train_nbin.astype("float64")

ms = sp.MinMaxScaler()

ms.fit(train_nbin_min)

train_nbin_min = pd.DataFrame(ms.transform(train_nbin_min),index=marge_index)

train_nbin_min.columns = col_list
#標準化

#train = train_scale

#正規化

#train = train_nbin_min

#正規化（binningあり）

train = train_min.drop(["Age","Fare"],axis=1)
test = train.loc[test_index.min():test_index.max(),:]

train = train.loc[train_index.min():train_index.max(),:]
split_train,split_test = train_test_split(train, test_size=0.3)
train_x = split_train.drop("Survived",axis=1)

train_y = pd.DataFrame(split_train.loc[:,"Survived"])

test_x = split_test.drop("Survived",axis=1)

test_y = pd.DataFrame(split_test.loc[:,"Survived"])
test = test.drop("Survived",axis=1)

train_y = pd.DataFrame(train.loc[:,"Survived"])

train_x = train.drop("Survived",axis=1)
"""

param_test = {

#    'subsample':[i/10.0 for i in range(6,10)],

#    'colsample_bytree':[i/10.0 for i in range(6,10)]}

#    'gamma':[i/10.0 for i in range(6,10)]}

#   'learning_rate':[i/10.0 for i in range(0,5)]}

#   'n_estimators':list(range(50,500,5))}

#   'max_depth':[4,5,6],

#   'min_child_weight':[2,3,4]}

#   'max_depth':list(range(1,10,2)),

#   'min_child_weight':list(range(1,10,2))}



grid_inst = GridSearchCV(estimator = xgb.XGBClassifier(

    learning_rate=0.1,

    n_estimators=160,

    max_depth=2,

    min_child_weight=2,

    gamma=0,

    subsample=0.6,

    colsample_bytree=0.6,

    objective= 'binary:logistic',

    nthread=4,

    scale_pos_weight=1,

    seed=20), 

    param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

"""


grid_inst = xgb.XGBClassifier(

    learning_rate=0.1,

    n_estimators=100,

    max_depth=4,

    min_child_weight=3,

    gamma=0.9,

    subsample=0.6,

    colsample_bytree=0.6,

    objective= 'binary:logistic',

    nthread=4,

    scale_pos_weight=1,

    seed=25)



fit = grid_inst.fit(train_x,train_y["Survived"])
#features =train.columns[train.any(axis=1)]

#fscore = grid_inst.feature_importances_

#fscore

#train.shape

xgb.plot_importance(grid_inst)

#feature = fit.feature_importances_

#plt.figure(figsize=(10,10))

#plt.barh(fscore)
#print("スコア：",grid_inst.grid_scores_)

#print("パラメータ : " ,grid_inst.best_params_)

#print("ベスト : " ,grid_inst.best_score_)
test_result = grid_inst.predict(test)

submit_df = pd.DataFrame({"Survived":test_result}, dtype=int, index=test_index)

submit_df.to_csv('submission.csv')
# 予測

test_x_index = test_x.index

test_result = grid_inst.predict(test_x)

test_result_df = pd.DataFrame({"Survived":test_result},index=test_x.index)
TP_cnt = 0

FP_cnt = 0

TN_cnt = 0

FN_cnt = 0



for index,row in split_test.iterrows():

 if row["Survived"] == 1:

  if test_result_df.at[index,"Survived"] == 1:

   TP_cnt = TP_cnt + 1

  else:

   FP_cnt = FP_cnt + 1

 else:

  if test_result_df.at[index,"Survived"] == 0:

   TN_cnt = TN_cnt + 1

  else:

   FN_cnt = FN_cnt + 1



result_Acc = ((TP_cnt + TN_cnt) / (TP_cnt + FP_cnt + TN_cnt + FN_cnt)) * 100

result_Pre = ((TP_cnt) / (TP_cnt + FP_cnt)) * 100

result_Rec = ((TP_cnt) / (TP_cnt + FN_cnt)) * 100

result_F = (TP_cnt / (TP_cnt + (1 / 2) * (FP_cnt + FN_cnt))) * 100



'[I] 全体での的中率'

print('正解率 : ' ,round(result_Acc,1))

print('適合率 : ' ,round(result_Pre,1))

print('再現率 : ' ,round(result_Rec,1))

print('F値 : ' ,round(result_F,1))
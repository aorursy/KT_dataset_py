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
import csv as csv
import math
from numpy import *
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier ,GradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn import datasets
from sklearn.cross_validation import cross_val_score
from sklearn import linear_model
#header=0は？
train_df = pd.read_csv("../input/train.csv", header=0)
train_df = train_df[train_df["Embarked"].notnull()]
train_df
train_df["Gender"] = train_df["Sex"].map({"female": 0, "male": 1}).astype(int)
train_df["EmbarkedInt"] = train_df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)
age1_df = train_df[train_df.Age >= 0]   #Ageあり
age0_df = train_df[train_df["Age"].isnull()]    #Ageなし
age1_features = age1_df[["Pclass","Age","SibSp","Parch","Gender","EmbarkedInt"]]         #特徴量のデータ 
age1_labels   = age1_df["Survived"]          #特徴量に対する正解データ
age0_features = age0_df[["Pclass","SibSp","Parch","Gender","EmbarkedInt"]]         #特徴量のデータ 
age0_labels   = age0_df["Survived"]          #特徴量に対する正解データ
# Load test data, Convert "Sex" to be a dummy variable
test_df = pd.read_csv("../input/test.csv", header=0)
test_df = test_df[test_df["Embarked"].notnull()]
test_df["Gender"] = test_df["Sex"].map({"female": 0, "male": 1}).astype(int)
test_df["EmbarkedInt"] = test_df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)
age1_t_df = test_df[test_df.Age >= 0]   #Ageあり
age0_t_df = test_df[test_df["Age"].isnull()]    #Ageなし
# Copy test data's "PassengerId" column, and remove un-used columns
ids_age1 = age1_t_df["PassengerId"].values
ids_age0 = age0_t_df["PassengerId"].values
#train-data
age1_df = age1_df.drop(["Name", "Ticket", "Sex", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)
age0_df = age0_df.drop(["Name", "Ticket", "Age", "Sex", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)
#必要なもののみ残してある
age1_df
#test-data
age1_t_df = age1_t_df.drop(["Name", "Ticket", "Sex", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)
age0_t_df = age0_t_df.drop(["Name", "Ticket", "Age", "Sex", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)
train_data_age1 = age1_df.values
test_data_age1 = age1_t_df.values
train_data_age1
#5つぐらいのモデルを適当に試して一番CVスコアが良いモデルで予測
#age1は最もスコアの良かったGradientBoostingClassifierを使って予測
model_age1 = GradientBoostingClassifier(n_estimators=100)
#1列目以降が特徴量、0列目がターゲット変数
output_age1 = model_age1.fit(train_data_age1[0::, 1::], train_data_age1[0::, 0]).predict(test_data_age1).astype(int)
#age0は最もスコアの良かったAdaBoostClassifierを使って予測
train_data_age0 = age0_df.values
test_data_age0 = age0_t_df.values
model_age0 = AdaBoostClassifier(n_estimators=50)
output_age0 = model_age0.fit(train_data_age0[0::, 1::], train_data_age0[0::, 0]).predict(test_data_age0).astype(int)
# export result to be "titanic_submit.csv"
submit_file = open("titanic_submit_r5.csv", "w", newline="")
file_object = csv.writer(submit_file)
file_object.writerow(["PassengerId", "Survived"])
file_object.writerows(zip(ids_age1, output_age1))
file_object.writerows(zip(ids_age0, output_age0))
submit_file.close()
train = train[train["Embarked"].notnull()]
test['Fare'].fillna(test['Fare'].median(), inplace = True)
#辞書ではなくリストで集計方法を渡すと、すべてのカラムの集計値が出る
print(train.groupby(["Pclass"]).agg(["count","mean"]))
#survived列に対してのみ集計値を取得
print(train.groupby(["Pclass"]).agg(["count","mean"])["Survived"])
print(train.groupby(["Sex"]).agg(["count","mean"])["Survived"])
#すべての列に対してcountされる
print(train.groupby(["Sex","Pclass"],as_index=False).count())
# (全行、sex,pclass,survived列)を取得
print(train.groupby(["Sex","Pclass"],as_index=False).count().loc[:, ["Sex","Pclass","Survived"]])
#Survivedをcountに変更、ならPassengerIdをcountにしても同じことではある
print(train.groupby(["Sex","Pclass"],as_index=False).count().loc[:, ["Sex","Pclass","Survived"]].rename(columns={"Survived":"count"}))
#groupbyにリストを渡すと組み合わせて分類してくれる
count = train.groupby(["Sex","Pclass"],as_index=False).count().loc[:, ["Sex","Pclass","Survived"]].rename(columns={"Survived":"count"})
#meanで生きてた人の割合になる
mean = train.groupby(["Sex","Pclass"],as_index=False).mean().loc[:, ["Sex","Pclass","Survived"]].rename(columns={"Survived":"ratio"})
rule = pd.merge(count, mean, on=["Sex","Pclass"])
print(rule)
passenger_id = list()
survived = list()
# 女性でpclassが1か2       →生存 
# 男性でpclassが2か３の人→死亡
for i in range(len(test)):
    data = test.iloc[i, :]
    if data["Sex"]=="female" and data["Pclass"]<=2:
        passenger_id.append(data["PassengerId"])
        survived.append(1)
    elif data["Sex"]=="male" and data["Pclass"]>=2:
        passenger_id.append(data["PassengerId"])
        survived.append(0)
#列方向で結合
output_df = pd.concat([pd.Series(passenger_id),pd.Series(survived)],axis=1)
output_df.columns = ["PassengerId", "Survived"]
#学習データとテストデータを先に結合しておくことで同時にcleaningが可能
data_cleaner = [train, test]
print(data_cleaner)
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
for dataset in data_cleaner:
    #特徴量:同乗家族人数(自身も含む)を生成し追加
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1    
    #特徴量:単独乗船フラグを生成し追加
    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
    #特徴量:敬称を生成し追加
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    #下記Cabin_Flagを追加
    #Cabinに値が入っているなら1,そうでないなら0
    dataset["Cabin_Flag"] = dataset["Cabin"].notnull().replace({True:1, False:0})

    #特徴量:Fareのbinをqcutで指定し追加(qcut:境界を自動的に設定し分類)
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

    dataset["Sex_Code"] = label.fit_transform(dataset['Sex'])
    dataset["Embarked_Code"] = label.fit_transform(dataset['Embarked'])
dataset['FareBin_Code']
#4分位数の境界値が入る
dataset['FareBin']
#特殊な敬称をクリーニング
stat_min = 10
title_names_train = (train['Title'].value_counts() < stat_min)
title_names_train
title_names_test = (test['Title'].value_counts() < stat_min)
#敬称の使用数が9以下のものはMiscと分類
train['Title'] = train['Title'].apply(lambda x: 'Misc' if title_names_train.loc[x] == True else x)
test['Title'] = test['Title'].apply(lambda x: 'Misc' if title_names_test.loc[x] == True else x)
train_age1 = train[train["Age"].notnull()]
test_age1 = test[test["Age"].notnull()]
data_cleaner_age1 = [train_age1, test_age1]
data_cleaner_age1
for dataset in data_cleaner_age1:
    #量ではなく値で分類。qcutは全体に対する量で等分割するが、cutは絶対値の等分割となる
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
train_age1 = train_age1.loc[:, ["PassengerId", "AgeBin_Code"]]
test_age1 = test_age1.loc[:, ["PassengerId", "AgeBin_Code"]]
#PassengerIdをマージ列名として、左側のテーブルに存在する（年齢がnullでない者）のみ、AgeBin_Code列をマージ（左外部結合）
train = pd.merge(train, train_age1, on="PassengerId", how="left")
test = pd.merge(test, test_age1, on="PassengerId", how="left")
#分類がめんどくさい女性かつpclass3,男性かつpclass1のデータのみ取得
train_rf = train[((train["Sex"]=="female")&(train["Pclass"]==3))|((train["Sex"]=="male")&(train["Pclass"]==1))]
test_rf = test[((test["Sex"]=="female")&(test["Pclass"]==3))|((test["Sex"]=="male")&(test["Pclass"]==1))]
#print(test_rf.groupby(["Sex","Pclass"],as_index=False).mean())
train_age1_rf = train_rf[train_rf["Age"].notnull()] #Ageあり
train_age0_rf = train_rf[train_rf["Age"].isnull()]  #Ageなし
test_age1_rf = test_rf[test_rf["Age"].notnull()]    #Ageあり
test_age0_rf = test_rf[test_rf["Age"].isnull()] #Ageなし
ids_age1 = list(test_age1_rf["PassengerId"])
ids_age0 = list(test_age0_rf["PassengerId"])
#train-dataをnumpy arrayとして用意
train_data_age1 = train_age1_rf.loc[:, ['Survived', 'Pclass', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Title_Code', 
                                      'Cabin_Flag', 'FareBin_Code', 'Sex_Code', 'Embarked_Code', 'AgeBin_Code']].values
#test-data
test_data_age1 = test_age1_rf.loc[:, ['Pclass', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Title_Code', 
                                      'Cabin_Flag', 'FareBin_Code', 'Sex_Code', 'Embarked_Code', 'AgeBin_Code']].values
train_data_age0 = train_age0_rf.loc[:, ['Survived', 'Pclass', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Title_Code', 
                                      'Cabin_Flag', 'FareBin_Code', 'Sex_Code', 'Embarked_Code']].values
test_data_age0 = test_age0_rf.loc[:, ['Pclass', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Title_Code', 
                                      'Cabin_Flag', 'FareBin_Code', 'Sex_Code', 'Embarked_Code']].values
xs_1 = train_data_age1[0::, 1::]
y_1 = train_data_age1[0::, 0]

xs_0 = train_data_age0[0::, 1::]
y_0 = train_data_age0[0::, 0]
xs_test1 = test_data_age1
xs_test0 = test_data_age0
parameters = {'max_depth': [2,4,6,8,10], 'n_estimators': [50,100,200]}
from xgboost import XGBClassifier
from sklearn import grid_search
clf = grid_search.GridSearchCV(XGBClassifier(), parameters)
clf.fit(xs_1, y_1)

print(clf.best_score_)
clf.best_params_
#引数で渡す時の**がポイントです。このようにすることで、辞書形式で関数に引数を渡すことができます。
clf_final = XGBClassifier(**clf.best_params_)
clf_final.fit(xs_1, y_1)
Y_pred1 = clf.predict(xs_test1).astype(int)
clf = grid_search.GridSearchCV(XGBClassifier(), parameters)
clf.fit(xs_0, y_0)
print(clf.best_score_)
clf_final = XGBClassifier(**clf.best_params_)
clf_final.fit(xs_0, y_0)
Y_pred0 = clf.predict(xs_test0).astype(int)
ids = pd.Series(ids_age1 + ids_age0)
pred = pd.Series(list(Y_pred1)+list(Y_pred0))
output_df2 = pd.concat([pd.Series(ids),pd.Series(pred)],axis=1)
output_df2.columns = ["PassengerId", "Survived"]
#output_df は面倒くさくない人たちのレコード（女性でpclass1,2あるいは男性でpclass3の人）
print(pd.concat([output_df,output_df2],axis=0))
print(pd.concat([output_df,output_df2],axis=0).sort_values(by="PassengerId"))
#drop=tureでないと旧インデックスがデータ列に残存してしまう
print(pd.concat([output_df,output_df2],axis=0).sort_values(by="PassengerId").reset_index(drop=True))
final_output = pd.concat([output_df,output_df2],axis=0).sort_values(by="PassengerId").reset_index(drop=True)
final_output.to_csv("predict_hybrid_r1.csv",index=False)












































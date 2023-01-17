import pandas as pd

import seaborn as sns



train = pd.read_csv("/kaggle/input/titanic/train.csv")

print(train)

test = pd.read_csv("/kaggle/input/titanic/test.csv")

print(test)



submit = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

submit.to_csv("gender_submission.csv",index=False)

print(submit)# kaggleが提供したデータを読み込んで表示。



# Seabornのリンクを確認するといろんな視覚化の例があるので、

# 必要だと思うAPIを使う

print(sns.countplot(data=train, x="Sex", hue="Survived"))



#　年齢と料金の関連性を確認

print(sns.lmplot(data=train, x="Age", y="Fare", hue="Survived", fit_reg=False))



#200以上の生存率は少ないため、　100以下のデータで行う

#その後、sns.lmplotでまた確認する(方法がわからないので放置)

low_fare = train[train["Fare"] < 100]



# Sex_encodeというカラムに性別を数字で格納

train.loc[train["Sex"] == "male", "Sex_encode"] = 0

train.loc[train["Sex"] == "female", "Sex_encode"] = 1

print(train.shape)

print(train[["Sex", "Sex_encode"]].head())



test.loc[test["Sex"] == "male", "Sex_encode"] = 0

test.loc[test["Sex"] == "female", "Sex_encode"] = 1

test.shape

test[["Sex", "Sex_encode"]].head()



#両方確認したら、test.csvのみ空になっていた

train[train["Fare"].isnull()]

test[test["Fare"].isnull()]



#NaNに値を埋めるために、Fare_fillinカラムを作成

train["Fare_fillin"] = train["Fare"]

#test.csvにも適用する

test["Fare_fillin"] = test["Fare"]



# FareがNaNになっている乗客を検索後、メジアンを入れる

test.loc[test["Fare"].isnull(), "Fare_fillin"] = test["Fare"].mode

test["Fare_fillin"]= 7.8292



# メジアンが入っているか確認

print(test.loc[test["Fare"].isnull(), ["Fare", "Fare_fillin"]])



#Age　２０歳以下を学生とする

train["Student"] = train["Age"] < 30

print(train.shape)

train[["Age", "Student"]].head(5)



#test.csvにも適用する

test["Student"] = test["Age"] < 30

print(test.shape)

test[["Age", "Student"]].head(5)
label_name = "Survived"

label_name

# feature_namesにFeatureとして使うカラムを格納

feature_names = [ "Sex_encode", "Fare_fillin", "Student"]



# feature_namesでtrainデータのfeatureを取得

X_train = train[feature_names]

print(X_train.shape)

X_train.head()



#test.csvにも適用する

X_test = test[feature_names]

# label_nameでtrainデータのlabelを取得

y_train = train[label_name]



import os

import numpy as np

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3,random_state=0,splitter='best')



model.fit(X_train,y_train)

predictions = model.predict(X_test)

print(X_test,predictions)





PassengerId=np.array(test["PassengerId"].astype(int))

gender_submission = pd.DataFrame({ 'Survived': predictions,'PassengerId':PassengerId })

gender_submission.to_csv("gender_submission.csv", index=False)



for dirname, _, filenames in os.walk('./'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
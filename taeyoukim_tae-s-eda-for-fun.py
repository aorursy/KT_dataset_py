import pandas as pd
train = pd.read_csv("../input/train.csv", index_col="PassengerId")



print(train.shape)



train.head()
test = pd.read_csv("../input/test.csv", index_col="PassengerId")



print(test.shape)



test.head()
%matplotlib inline



import seaborn as sns



import matplotlib.pyplot as plt
#1) 남성 생존자, 2) 남성 사망자, 3) 여성 생존자, 4) 여성 사망자 를 시각화!

sns.countplot(data=train, x="Sex", hue="Survived")
#성별(Sex) 컬럼과 마찬가지로, 객실 등급(Pclass) 컬럼도 간단하게 분석해볼까요

sns.countplot(data=train, x="Pclass", hue="Survived")
# 선착장(Embarked) 컬름을 시각화해봅니다

sns.countplot(data=train, x="Embarked", hue="Survived")
# pivot_table을 통해 피벗으로 한번 표시해볼까요. 생존률!

pd.pivot_table(train, index="Embarked", values="Survived")
sns.lmplot(data=train, x="Age", y="Fare", hue="Survived", fit_reg=False) ##리그레션 Line은 종종 정확도가 떨어지기때문에... 끄겠습니다. False!
low_fare = train[train["Fare"] < 500]

train.shape, low_fare.shape
## 500이하 Low Fare를 기준으로 다시 lmplot을 그립니다

sns.lmplot(data=low_fare, x="Age", y="Fare", hue="Survived", fit_reg=False)
## * 50달러 구간 100달러 구간, 200달러 이상 구간을 기준으로 생존자와 사망자의 비율이 크게 차이나기 시작하며 운임요금과 나이간의 특정한 상관관계는.. 없다.;; ㅡ,.ㅡ;
low_low_fare = train[train["Fare"] < 100]

train.shape, low_fare.shape, low_low_fare.shape
sns.lmplot(data=low_low_fare, x="Age", y="Fare", hue="Survived", fit_reg=False)
# 가족 수를 셀 때는 언제나 나 자신도 1 포함.. 나 자신은 SibSp와 Parch 중 어디에도 포함되어 있지 않기 때문에, 무조건 1을 더해서 가즈아..

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1



print(train.shape)



train[["SibSp", "Parch", "FamilySize"]].head()
# 위에서 정의한 가족 수(FamilySize)를 그려봅니다

sns.countplot(data=train, x="FamilySize", hue="Survived")
train.loc[train["FamilySize"] == 1, "FamilyType"] = "Single"



train.loc[(train["FamilySize"] > 1) & (train["FamilySize"] < 5), "FamilyType"] = "Nuclear"



train.loc[train["FamilySize"] >= 5, "FamilyType"] = "Big"



print(train.shape)



train[["FamilySize", "FamilyType"]].head(10)
sns.countplot(data=train, x="FamilyType", hue="Survived")
pd.pivot_table(data=train, index="FamilyType", values="Survived")
train["Name"].head()
# 함수정의를 통해 이름을 받았을 때 이름에서 타이틀을 반환해주는...

def get_title(name):

    # 먼저 name을 , 을 기준으로 쪼갭니다. 쪼갠 결과는 0) Braund와 1) Mr. Owen Harris가 됩니다.

    # 여기서 1)번을 가져온 뒤 다시 . 을 기준으로 쪼갭니다. 쪼갠 결과는 0) Mr와 1) Owen Harris가 됩니다.

    # 여기서 0)번을 반환합니다. 최종적으로는 Mr를 반환하게 됩니다.

    return name.split(", ")[1].split('. ')[0]

train["Name"].apply(get_title).unique()
# 작위를 저장하는 컬럼은 없으므로 "Title"이라는 새로운 컬럼을 만듭니다.

train.loc[train["Name"].str.contains("Mr"), "Title"] = "Mr"



train.loc[train["Name"].str.contains("Miss"), "Title"] = "Miss"



train.loc[train["Name"].str.contains("Mrs"), "Title"] = "Mrs"



train.loc[train["Name"].str.contains("Master"), "Title"] = "Master"



print(train.shape)



train[["Name", "Title"]].head(10)
## 위에 적용한 전처리내용을 그립니다..

sns.countplot(data=train, x="Title", hue="Survived")
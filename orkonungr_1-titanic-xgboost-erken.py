import xgboost as xgb

import numpy as np

import pandas as pd
def tranform(df):

    df["Sex"] = df["Sex"].replace({"male":1,"female":0})

    df["Embarked"] = df["Embarked"].replace({"Q":1,"S":0,"C":0})

    df["Name_length"] = df["Name"].apply(len)

    df["Ticket"] = df["Ticket"].apply(len)

    df["Cabin"] = df["Cabin"].apply(str).apply(len)

    df["Age"] = df["Age"].fillna(30)

    cols = ["Pclass", "Sex", "Age", "SibSp", "Parch",

    "Ticket", "Fare", "Cabin", "Embarked", "Name_length"]

    df = df[cols]

    return df
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
y = train["Survived"]

submission=pd.DataFrame(test["PassengerId"])

X = tranform(train)

test = tranform(test)
model = xgb.XGBClassifier(max_depth=5,

eta=0.05, silent=False, reg_lambda=0.5,

objective="binary:logistic", n_estimators=10)

model.fit(X,y,eval_metric=["error"])

subs = model.predict(test)

submission["Survived"] = (subs>0.5).astype(int)

submission.to_csv("sub3.csv",index=False)
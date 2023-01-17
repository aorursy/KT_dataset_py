# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv("/kaggle/input/titanic/test.csv")

test_passenger_id = test["PassengerId"]
features=["Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]

nan_values=[]

for i in features:

    nan_value=(np.sum(train[i].index.value_counts()))-(np.sum(train[i].value_counts()))

    nan_values.append(nan_value)

nan_values
train.drop(["Cabin","Ticket","Name"],axis=1,inplace=True)
train.columns
train.Embarked.value_counts()
train.Embarked=[0 if i=="S" else 1 if i=="C" else 2 for i in train.Embarked]
train.Embarked.value_counts()
train.Sex.value_counts()
train.Sex=[1 if i=="male" else 0 for i in train.Sex]
train.Sex.value_counts()
train.info()
train.Sex.value_counts().index
def bar_plot(feature):

    sns.set(style="white", context="talk")

    plt.subplots(figsize=(6,3))

    sns.barplot(x=train[feature].value_counts().index,y=train[feature].value_counts())

    plt.show()
feature=["Survived","Pclass","Sex","SibSp","Parch","Embarked"]

for i in feature:

    bar_plot(i)
train.columns
# Pclass - Survived

# Sex - Survived

# SibSp - Survived

# Parch - Survived

print(train[["Pclass","Survived"]].groupby(["Pclass"]).mean().sort_values(by=("Survived")))

print(train[["Sex","Survived"]].groupby(["Sex"]).mean().sort_values(by=("Survived")))

print(train[["SibSp","Survived"]].groupby(["SibSp"]).mean().sort_values(by=("Survived")))

print(train[["Parch","Survived"]].groupby(["Parch"]).mean().sort_values(by=("Survived")))

train.drop(["Age"],axis=1,inplace=True)
train.isnull().sum()
test.drop(["Age","Cabin","Name","Ticket"],axis=1,inplace=True)
test.isnull().sum()
test[test["Fare"].isnull()]
test["Fare"]=test["Fare"].fillna(np.mean(test[test["Pclass"]==3]["Fare"]))
test[test["Fare"].isnull()]
train.columns
feature_heatmap=["Survived","Pclass","Sex","SibSp","Parch","Fare","Embarked"]

sns.heatmap(train[feature_heatmap].corr(),annot=True,fmt=".2f")

plt.show()
train["Family"]=train["SibSp"]+train["Parch"]+1
test["Family"]=test["SibSp"]+test["Parch"]+1
test["Sex"]=[1 if i=="male" else 0 for i in test["Sex"]]
test.head()
train.head()
plt.subplots(figsize=(9,6))

sns.barplot(x=train["Family"],y=train["Survived"])

plt.show()
train["Family_size"]=[0 if i == 1 else 1 if i < 5 else 2 for i in train["Family"]]
test["Family_size"]=[0 if i == 1 else 1 if i < 5 else 2 for i in test["Family"]]
train.head(30)
train["Family_size"].value_counts()
plt.subplots(figsize=(9,6))

sns.barplot(x=train["Family_size"],y=train["Survived"])

plt.show()
train=pd.get_dummies(train,columns=["Family_size"])

train.head()
test=pd.get_dummies(test,columns=["Family_size"])
train = pd.get_dummies(train, columns=["Embarked"])

train.head()
test = pd.get_dummies(test, columns=["Embarked"])
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
x_train_df=train.drop(["Survived"],axis=1)

y_train_df=train["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x_train_df, y_train_df, test_size = 0.3, random_state = 42)

print("x_train",len(x_train))

print("x_test",len(x_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
leg=LogisticRegression()

leg.fit(x_train,y_train)

print("score",leg.score(x_train,y_train))
modeller=[LogisticRegression(random_state=42),KNeighborsClassifier()]

leg_param_grid={"C":[0.001,.009,0.01,.09,1,5,10,25],"penalty":["l1","l2"]}

knn_param_grid={"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

model_params=[leg_param_grid,knn_param_grid]
accuracy=[]

clf_param=[]

clf_estimator=[]

for i in range(len(modeller)):

    clf=GridSearchCV(modeller[i],param_grid=model_params[i],cv=StratifiedKFold(n_splits = 10),n_jobs=-1,scoring = "accuracy")

    clf.fit(x_train,y_train)

    accuracy.append(clf.best_score_)

    clf_param.append(clf.best_params_)

    clf_estimator.append(clf.best_estimator_)

    print("accuray",accuracy[i])

    print("parametreler",clf_param[i])

    print("estimator",clf_estimator[i])
votingC = VotingClassifier(estimators = [("lr",clf_estimator[0]),

                                        ("knn",clf_estimator[1])],

                                        voting = "soft")

votingC = votingC.fit(x_train, y_train)

print(accuracy_score(votingC.predict(x_test),y_test))
test.head()
predict_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

survived = pd.concat([test_passenger_id, predict_survived],axis = 1)

survived.to_csv("titanic_survived.csv", index = False)
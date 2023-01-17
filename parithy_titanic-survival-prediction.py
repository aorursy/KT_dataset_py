# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
titanic = pd.read_csv("../input/train.csv")

print(titanic.columns)



columns = ["PassengerId","Survived" ,"Pclass","Sex" ,"Age","SibSp", "Parch", "Fare" ,"Embarked"]

titanic=titanic[columns]

print(titanic.head())
titanic["Age"]= titanic["Age"].fillna(titanic["Age"].median())

print(titanic.describe())



titanic.loc[titanic["Sex"] == "male", "Sex"] =0

titanic.loc[titanic["Sex"] == "female", "Sex"] =1

print(titanic["Embarked"].unique())

titanic["Embarked"]=titanic["Embarked"].fillna('S')

print(titanic.head())

titanic.loc[titanic["Embarked"] == "S" , "Embarked"] = 0

titanic.loc[titanic["Embarked"] == "C" , "Embarked"] = 1

titanic.loc[titanic["Embarked"] == "Q" , "Embarked"] = 2

print(titanic["Embarked"].unique())
from sklearn import linear_model

from sklearn import cross_validation



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]



X_train = titanic[predictors]

Y_train = titanic["Survived"]

clf = linear_model.LinearRegression()

clf.fit(X_train,Y_train)
prdata = pd.read_csv("../input/test.csv")

prdata = prdata[predictors]

prdata["Age"]= prdata["Age"].fillna(prdata["Age"].median())

print(prdata.describe())



prdata.loc[prdata["Sex"] == "male", "Sex"] =0

prdata.loc[prdata["Sex"] == "female", "Sex"] =1



prdata["Embarked"]=prdata["Embarked"].fillna('S')

#prdata["Embarked"]=prdata["Embarked"].dropna()

print(prdata["Embarked"].unique())



prdata.loc[prdata["Embarked"] == "S" , "Embarked"] = 0

prdata.loc[prdata["Embarked"] == "C" , "Embarked"] = 1

prdata.loc[prdata["Embarked"] == "Q" , "Embarked"] = 2

print(prdata.describe())



prdata1=pd.DataFrame()



print(prdata1.columns)

prdata["Survived"] = clf.predict(prdata)

print(clf.coef_,clf.intercept_)

scores = cross_validation.cross_val_score(clf, prdata[predictors], prdata["Survived"], cv=3)

#prdata.loc[prdata["Survived"] < 0.5 , "Survived"] = 0

#prdata.loc[prdata["Survived"] > 0.5 , "Survived"] = 1

prdata["Survived"]=prdata["Survived"].apply(np.round).astype(int)
prdata1 = prdata[predictors]

#prdata1= prdata1.drop(["Survived"], axis=1)

from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train,Y_train)

prdata1["Survived"]= rfc.predict(prdata1)

print(prdata1.head())

print(prdata.head())



Lr_count = prdata.groupby(prdata1["Sex"]).count()

rf_count = prdata1.groupby(prdata1["Sex"]).count()

print(Lr_count["Survived"])

print(rf_count["Survived"])

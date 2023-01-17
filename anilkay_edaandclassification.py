# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

%matplotlib inline

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/train.csv")

data.head()
import seaborn as sns
data.groupby(data["Pclass"]).count()["Survived"]
data["Survived"].groupby(data["Survived"]).count()
sns.countplot(data=data,x="Sex")
data["Age"].describe()
sns.heatmap(data.corr(),annot = True)
survive=data[data["Survived"]==1]

survive.head()
sns.countplot(data=survive,x="Sex")
sns.countplot(data=survive,x="Pclass")
survive["Age"].median()
survive["Age"].mean()
survive["Age"].max()
dead=data[data["Survived"]==0]

dead.head()
sns.countplot(data=dead,x="Sex")
dead["Age"].median()
dead["Age"].mean()
dead["Age"].max()
dead["Age"].min()
sns.stripplot(x="Age", y="Sex", data=data,hue="Survived")
sns.stripplot(x="Pclass", y="Fare", data=data,hue="Survived")
len(set(data["Ticket"]))
data.shape
classificationdata=data[["Pclass","Sex","Age","SibSp","Parch","Embarked","Survived"]]

classificationdata.head()
classificationdata.shape
classificationdata["Sex"]=classificationdata["Sex"].map({"female":1,"male":-1})

classificationdata["Embarked"]=classificationdata["Embarked"].map({'C':1, 'Q':2, 'S':3})

from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

classificationdata=imp_mean.fit_transform(classificationdata)

classificationdata=pd.DataFrame(classificationdata)

x=classificationdata.iloc[:,0:6]

y=classificationdata.iloc[:,6:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(x,y,test_size=0.20,random_state=23)
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()

dtc.fit(X_train,y_train)

ypred=dtc.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from sklearn.ensemble import RandomForestClassifier

dtc=RandomForestClassifier()

dtc.fit(X_train,y_train)

ypred=dtc.predict(X_test)

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
a=classificationdata["Sex"].map({"female":1,"male":-1})

a.head()
set(classificationdata["Embarked"])
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_mean.fit_transform(classificationdata)
from sklearn.svm import SVC

dtc=SVC(kernel="rbf",gamma="auto",decision_function_shape="ovo")

dtc.fit(X_train,y_train)

ypred=dtc.predict(X_test)

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from sklearn.svm import SVC

dtc=SVC(kernel="rbf",gamma="auto",decision_function_shape="ova")

dtc.fit(X_train,y_train)

ypred=dtc.predict(X_test)

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from sklearn.svm import SVC

dtc=SVC(kernel="sigmoid",gamma="auto",decision_function_shape="ova",probability=True)

dtc.fit(X_train,y_train)

ypred=dtc.predict(X_test)

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from sklearn.svm import SVC

dtc=SVC(kernel="rbf",gamma=2,decision_function_shape="ovr",probability=True)

dtc.fit(X_train,y_train)

ypred=dtc.predict(X_test)

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
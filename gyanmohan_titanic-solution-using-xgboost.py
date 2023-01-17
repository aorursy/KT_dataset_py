# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt 

import math
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.shape

test.shape



train.describe()

test.describe()



train_1.isna().sum()

test.isna().sum()

train["family_size"]=train.SibSp+train.Parch+1

test["family_size"] = test.SibSp + test.Parch + 1
train_1 = train.drop(["Ticket","Name","SibSp","Parch","Cabin"],axis=1)

test_1 =test.drop(["Ticket","Name","SibSp","Parch","Cabin"],axis=1)
table= pd.crosstab(train_1.Sex,train_1.Survived)
train_1.groupby("Sex").Survived.mean()

train_1.groupby("Pclass").Survived.mean()

train_1.groupby(["Sex","Pclass"]).Survived.mean()

train_1.groupby(["Pclass","Sex"]).mean()

train_1.groupby(["Pclass","Sex"]).mean()["Survived"].plot.bar()
def bar_chart(features):

    survive=train_1[train_1.Survived==1][features].value_counts()

    dead =train_1[train_1.Survived==0][features].value_counts()

    df = pd.DataFrame([survive,dead])

    df.index = ["survived","dead"]

    df.plot(kind="bar",stacked=True,figsize=(10,5))

bar_chart("Sex")

bar_chart("Pclass")

bar_chart("family_size")

bar_chart("Age")

bar_chart("Embarked")   
facet = sns.FacetGrid(train_1,hue="Survived",aspect=4)

facet.map(sns.kdeplot,"Age",shade=True)

facet.add_legend

 

facet = sns.FacetGrid(train_1,hue="Survived",aspect=4)

facet.map(sns.kdeplot,"Sex",shade=True)

facet.add_legend



facet = sns.FacetGrid(train_1,hue="Survived",aspect=4)

facet.map(sns.kdeplot,"family_size",shade=True)

facet.add_legend



facet = sns.FacetGrid(train_1,hue="Survived",aspect=4)

facet.map(sns.kdeplot,"Pclass",shade=True)

facet.add_legend
Pclass1 = train_1[train_1.Pclass==1]["Embarked"].value_counts()

Pclass2 = train_1[train_1.Pclass==2]["Embarked"].value_counts()

Pclass3 = train_1[train_1.Pclass==3]["Embarked"].value_counts()

df= pd.DataFrame([Pclass1,Pclass2,Pclass3])

df.index = ["1stclass","2ndclass","3rdclass"]

df.plot(kind="bar",stacked=True)

sns.heatmap(train_1.isna())

sns.heatmap(test_1.isna())

sns.heatmap(train_1.corr())



train_1["Age"].fillna(train_1.Age.mean(),inplace=True)

test_1["Age"].fillna(test_1.Age.mean(),inplace=True)

test_1["Fare"].fillna(test_1.Fare.mean(),inplace=True)
common_value = "S"

data = [train_1]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

sns.heatmap(train_1.isnull(),cmap="viridis")
PC1 = pd.get_dummies(train_1.Sex,drop_first=True)

PC2 = pd.get_dummies(train_1.Pclass,drop_first=True)

PC3 = pd.get_dummies(train_1.Embarked,drop_first=True)

train_1 = pd.concat([train_1,PC1,PC2,PC3],axis=1,)

train_1 = train_1.drop(["Sex","Pclass","Embarked"],axis=1)





X_train = train_1.drop(["Survived"],axis=1)

Y_train = train_1.Survived



PC4 = pd.get_dummies(test_1.Sex,drop_first=True)

PC5 = pd.get_dummies(test_1.Pclass,drop_first=True)

PC6 = pd.get_dummies(test_1.Embarked,drop_first=True)

test_1=pd.concat([test_1,PC4,PC5,PC6],axis=1)

X_test = test_1.drop(["Sex","Pclass","Embarked"],axis=1)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

logmodel = LogisticRegression()

logmodel.fit(X_train,Y_train)

train_pred_lr=logmodel.predict(X_train)

accuracy_score(train_pred_lr,Y_train)

confusion_matrix(train_pred_lr,Y_train)

classification_report(train_pred_lr,Y_train)

test_pred_lr = logmodel.predict(X_test)



result_lr = pd.concat([X_test.PassengerId],axis=1)

result_lr["Survived"]=test_pred_lr
from sklearn.ensemble import AdaBoostClassifier 

model_log = logmodel

Adaboost_log = AdaBoostClassifier(base_estimator=model_log,n_estimators=400,learning_rate=1)    

boostmodel_log = Adaboost_log.fit(X_train,Y_train)

train_boost_pred_log=boostmodel_log.predict(X_train)

test_boost_pred_log = boostmodel_log.predict(X_test)

accuracy_score(train_boost_pred_log,Y_train)



result_boost = pd.concat([X_test.PassengerId],axis=1)

result_boost["Survived"] = test_boost_pred_log
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

K_fold = KFold(n_splits=10,shuffle=True,random_state=True)



dtc=DecisionTreeClassifier()

scoring="accuracy"

score = cross_val_score(dtc,X_train,Y_train,cv=K_fold,n_jobs=1,scoring=scoring)

print(score)

round(np.mean(score)*100,2)

dtc_2 = DecisionTreeClassifier(criterion="entropy",max_depth=1)

Adaboost = AdaBoostClassifier(base_estimator=dtc_2,n_estimators=400,learning_rate=1)

boostmodel = Adaboost.fit(X_train,Y_train)

boost_pred = boostmodel.predict(X_test)

train_boost_pred = boostmodel.predict(X_train)

accuracy_score(Y_train,train_boost_pred)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=15)

scoring="accuracy"

score = cross_val_score(rf,X_train,Y_train,cv=K_fold,n_jobs=1,scoring=scoring)

print(score)

round(np.mean(score)*100,2)
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV

params = {"learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],"max_depth":[1,2,3,4,5,6,8,9,10],"min_child_weight":[1,2,3,4,5,6,7,8,9],"gamma":[0.0,0.1,0.2,0.3,0.4,0.5],"colsample_bytree":[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],"n_estimators":[100,200,300,400,500]}

classifier = XGBClassifier()

random_search = RandomizedSearchCV(classifier,param_distributions=params,n_iter=10,scoring="roc_auc",n_jobs=-1,cv=5,verbose=3)

random_search.fit(X_train,Y_train)

random_search.best_estimator_    



XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.5, gamma=0.2,

              learning_rate=0.1, max_delta_step=0, max_depth=5,

              min_child_weight=1, missing=None, n_estimators=200, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)

XGB_fit = XGB.fit(X_train,Y_train)

pred_XGB = XGB.predict(X_test)

train_pred_XGB = XGB.predict(X_train)

#94%

confusion_matrix(train_pred_XGB,Y_train)

classification_report(train_pred_XGB,Y_train)



result_XGB = pd.concat([X_test.PassengerId],axis=1)

result_XGB["Survived"] = pred_XGB

result_XGB.to_csv("Submission_XGboost.csv",index=False)
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
train2=pd.read_csv("../input/titanic/train.csv")

test_=pd.read_csv("../input/titanic/test.csv")

train2.head(15)
train2.dtypes
train2.shape
train2.isnull().sum()
mean_age=round(train2['Age'].mean())

train2.fillna({"Age":mean_age,"Embarked":'Q'}, inplace=True)

train2.head(15)
train2.fillna({'Cabin':'None'},inplace=True)
train2.head()
convert_={'Fare': int,'Age': int}

train2=train2.astype(convert_)

train2.head()
train2["Sex"]=train2["Sex"].apply(lambda x:1 if x=="male" else 0)

train2.head()
convert_dict={"S":1,

             "C":2,

             "Q":3}

train2['embarked']=train2['Embarked'].map(convert_dict)

train2.head(10)
passenger=train2["PassengerId"]
train2.head()
x_train=train2.drop(['PassengerId','Name','Ticket','Embarked','Survived','Sex','Cabin'],axis=1)

x_train.head()
y_train=train2["Survived"]

y_train
x_train=pd.get_dummies(x_train)

x_train.head()
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(x_train, y_train , test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(C=1.75034884505077,penalty = 'l2',random_state=0).fit(X_train,Y_train)
Y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score

scores=accuracy_score(Y_test,Y_pred)

scores
from sklearn.tree import DecisionTreeClassifier

dlf=DecisionTreeClassifier(random_state=0).fit(X_train,Y_train)
y_=dlf.predict(X_test)
score=accuracy_score(Y_test,y_)

score
from xgboost import XGBClassifier

xlf=XGBClassifier(random_state=0).fit(X_train,Y_train)
y_p=xlf.predict(X_test)
sc_=accuracy_score(Y_test,y_p)

sc_
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)
y_knn=knn.predict(X_test)
from sklearn.metrics import accuracy_score

score_knn=accuracy_score(Y_test,y_knn)

score_knn
from sklearn.metrics import f1_score

best_k=0

best_score=0

neighbors=range(1,10,2)

for k in neighbors:

    kn=KNeighborsClassifier(n_neighbors=k).fit(X_train,Y_train)

    kn_pred=kn.predict(X_test)

    f1=f1_score(Y_test,kn_pred)

    if f1>best_score:

        best_k=k

        best_score=f1

knn=KNeighborsClassifier(n_neighbors=best_k)

knn.fit(X_train,Y_train)

best_knn_pred=knn.predict(X_test)

accuracy_score(Y_test,best_knn_pred)
from sklearn.ensemble import RandomForestClassifier

rlf=RandomForestClassifier(criterion= 'gini',max_depth = None,min_samples_leaf =5,min_samples_split =5,n_estimators =200 ,random_state=0).fit(X_train,Y_train)
y_rlf=rlf.predict(X_test)
score_rlf=accuracy_score(Y_test,y_rlf)

score_rlf
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(random_state=0).fit(X_train, Y_train)
y_boost=model.predict(X_test)

accuracy_score(Y_test,y_boost)
test_.head()
test_.isnull().sum()
mean_age=round(test_['Age'].mean())

mean_=round(test_['Fare'].mean())

test_.fillna({"Age":mean_age,"Fare":mean_}, inplace=True)

test_.fillna({'Cabin':'None'},inplace=True)
convert_={'Fare': int,'Age': int}

test_=test_.astype(convert_)

test_.head()
test_train=test_.drop(['PassengerId','Name','Sex','Ticket','Cabin'],axis=1)

test_train
convert_dict={"S":1,

             "C":2,

             "Q":3}

test_train['embarked']=test_train['Embarked'].map(convert_dict)

test_train.head(10)
test_train=test_train.drop(['Embarked'],axis=1)
test_train
rlf.fit(x_train,y_train)
predict=rlf.predict(test_train)
output = pd.DataFrame({'PassengerId': test_.PassengerId, 'Survived':predict})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
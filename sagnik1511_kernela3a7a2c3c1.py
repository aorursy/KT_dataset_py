#author s_agnik151

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname,_,filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname,filename))
trd=pd.read_csv("/kaggle/input/titanic/train.csv")

trd.head()
ted=pd.read_csv("/kaggle/input/titanic/test.csv")

ted.head()
w=trd.loc[trd.Sex=='female']["Survived"]

rw= sum(w)/len(w)

print("percentage of those who survived are women is : ",rw)
m=trd.loc[trd.Sex=='male']["Survived"]

rm=sum(m)/len(m)

print("percentage of those who survived are men is : ",rm)
#author s_agnik1511

from sklearn.ensemble import RandomForestClassifier

y=trd["Survived"]

features=["Pclass","Sex","SibSp","Parch"]

X=pd.get_dummies(trd[features])

X_test=pd.get_dummies(ted[features])

model=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)

model.fit(X,y)

predictions = model.predict(X_test)

output=pd.DataFrame({'PassengerId':ted.PassengerId,'Survived':predictions})

output.to_csv('s_agnik1511 first sub',index=False)

print("Your submission was successfully saved!")
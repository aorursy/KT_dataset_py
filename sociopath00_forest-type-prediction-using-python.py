import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



%matplotlib inline
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
list(train.columns.values)
#separating "ID" and response variable "Cover_Type"

Id=test['Id']

y=train['Cover_Type']
train.describe()
train.isnull().sum()
sns.countplot(data=train,x=train['Cover_Type'])
sns.boxplot(x="Cover_Type", y="Elevation", data=train);
sns.boxplot(x="Cover_Type", y="Aspect",data=train);
train=train.drop(['Id','Cover_Type'],1)

test=test.drop(['Id'],1)
x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.3, random_state=42)
rf=RandomForestClassifier(n_estimators=300,class_weight='balanced',n_jobs=2,random_state=42)
rf.fit(x_train,y_train)
pred=rf.predict(x_test)
acc=rf.score(x_test,y_test)

print(acc)
rf.fit(train,y)
ct=rf.predict(test)

print(ct)
output=pd.DataFrame(Id)

output['Cover_Type']=ct

output.head()
output.to_csv("output.csv",index=False)
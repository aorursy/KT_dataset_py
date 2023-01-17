# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
!pip install ppscore
!pip install lazypredict
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()
import ppscore as pps
pps.matrix(train)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
sns.set_style('whitegrid')
sns.countplot(x=train['Survived'],data=train,hue='Sex')
plt.show()
sns.countplot(x=train['Survived'],data=train,hue='Pclass')
plt.show()
sns.distplot(train['Age'].dropna(),kde=True,bins=30)
train.info()
sns.countplot(x='SibSp',data=train)
train["Fare"].hist(bins=70,figsize=(10,4))
plt.show()
sns.boxplot(x='Pclass',y='Age',data=train)
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age']=train[["Age","Pclass"]].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
train.drop('Cabin',inplace=True,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
train.dropna(inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
sex=pd.get_dummies(train['Sex'])
sex
sex=pd.get_dummies(train['Sex'],drop_first=True)
sex
embark =pd.get_dummies(train['Embarked'],drop_first=True)
embark
train=pd.concat([train,sex,embark],axis=1)
train.head()
import re
l=[]
x=0
for i in train["Name"]:
    if(str(i).find("Mr.")>0 or str(i).find("Mrs.")>0):
        l.append(1)
    else:
        l.append(0)
    print(l[x],i)
    x+=1
train['Maritial_Status'] = l
train.head()
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.head()
y=train["Survived"]
x=train.drop("Survived",axis=1)
import lazypredict
import sys
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=.4,random_state =23)
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(train)
scaled_features=scaler.transform(train)
train=pd.DataFrame(scaled_features,columns=train.columns)
train.head()
from sklearn.model_selection import train_test_split as tits
x_train,x_test,y_train,y_test=tits(x,y,test_size=0.2,random_state=23)
from sklearn.linear_model import LogisticRegression as lr
logmodel=lr()
logmodel.fit(x_train,y_train)
predictions =logmodel.predict(x_test)
from sklearn.metrics import classification_report as cr
print(cr(y_test,predictions))
test=pd.read_csv('/kaggle/input/titanic/test.csv')
test.isna().sum()
test['Age']=test[["Age","Pclass"]].apply(impute_age,axis=1)
test.drop('Cabin',inplace=True,axis=1)
sex=pd.get_dummies(test['Sex'])
sex
sex=pd.get_dummies(test['Sex'],drop_first=True)
sex
embark =pd.get_dummies(test['Embarked'],drop_first=True)
embark
test=pd.concat([test,sex,embark],axis=1)
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
import re
l=[]
xx=0
for i in test["Name"]:
    if(str(i).find("Mr.")>0 or str(i).find("Mrs.")>0):
        l.append(1)
    else:
        l.append(0)
    print(l[xx],i)
    xx+=1
test['Maritial_Status'] = l
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
x.head()
test.head()
from sklearn.ensemble import RandomForestClassifier
rdmf = RandomForestClassifier(n_estimators=100,max_depth=5, criterion='entropy')
rdmf.fit(x,y)
scaler.fit(test)
scaled_features=scaler.transform(test)
test=pd.DataFrame(scaled_features,columns=test.columns)
test.head()
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x,y)
test['Fare'] = test['Fare'].fillna((test['Fare'].mean()))
predictions =rdmf.predict(test)
ft=pd.read_csv('/kaggle/input/titanic/test.csv')
predi = pd.DataFrame(predictions, columns=['predictions'])
predi.head()
data = [ft["PassengerId"], predi["predictions"]]
headers = ["PassengerId", "Survived"]
final = pd. concat(data, axis=1, keys=headers)
final.to_csv("res1.csv",index=False)



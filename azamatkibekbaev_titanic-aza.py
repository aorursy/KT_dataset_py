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
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")
train.head()
train.isnull().sum()
import seaborn as sns
sns.countplot(train['Sex'])
sns.factorplot('Pclass',data=train,kind='count',hue='Sex')
def child(x):
    if x<12:
        return 'Child'
    else:
        return 'Elder'

train['Person']=train['Age'].apply(child)
train.head(10)
sns.factorplot('Pclass',data=train,kind='count',hue='Person')
fig=sns.FacetGrid(train,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
a=train['Age'].max()
fig.set(xlim=(0,a))
fig.add_legend()
fig=sns.FacetGrid(train,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
a=train['Age'].max()
fig.set(xlim=(0,a))
fig.add_legend()
sns.factorplot('Embarked',data=train,kind='count',hue='Pclass')
def fam(x):
    sb,p=x
    if (sb==0) & (p==0):
        return 'alone'
    else:
        return 'family'
    
train['Family']=train[['SibSp','Parch']].apply(fam,axis=1)
train.head()
sns.factorplot('Family',data=train,kind='count',hue='Pclass')
sns.countplot(train['Survived'])
p_null=(len(train)-train.count())*100/len(train)
p_null
df1=train.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
df1.head()
df1['Embarked'].fillna('S',inplace=True)
df1.isnull().any()
df1['Age'].interpolate(inplace=True)
df1.isnull().any()
df1 = pd.get_dummies(df1, columns=["Sex","Embarked","Person","Family"])
df1.head()
from sklearn.preprocessing import MinMaxScaler
a=MinMaxScaler()
scaled=a.fit_transform(df1[['Age','Fare']])
df1[['Age','Fare']]=pd.DataFrame(scaled)
df1.head()
df1.corr()
from sklearn.model_selection import train_test_split
X=df1.drop('Survived',axis=1)
y=df1['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
import xgboost as xgb
model1 = xgb.XGBClassifier()
model2 = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)

train_model1 = model1.fit(X_train, y_train)
train_model2 = model2.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
pred1 = train_model1.predict(X_test)
pred2 = train_model2.predict(X_test)
print("Accuracy for model 1: %.2f" % (accuracy_score(y_test, pred1) * 100))
print("Accuracy for model 2: %.2f" % (accuracy_score(y_test, pred2) * 100))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc_model = rfc.fit(X_train, y_train)
pred8 = rfc_model.predict(X_test)
print("Accuracy for Random Forest Model: %.2f" % (accuracy_score(y_test, pred8) * 100))
# Test set

def fam(x):
    sb,p=x
    if (sb==0) & (p==0):
        return 'alone'
    else:
        return 'family'
    
test['Family']=test[['SibSp','Parch']].apply(fam,axis=1)
test.head()
def child(x):
    if x<12:
        return 'Child'
    else:
        return 'Elder'

test['Person']=test['Age'].apply(child)
test.head(10)
df2=test.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
df2.head()
df2 = pd.get_dummies(df2, columns=["Sex","Embarked","Person","Family"])
df2.head()
from sklearn.preprocessing import MinMaxScaler
a=MinMaxScaler()
scaled=a.fit_transform(df2[['Age','Fare']])
df2[['Age','Fare']]=pd.DataFrame(scaled)
df2.head()
pred4 = train_model2.predict(df2)
pred4
pred=pd.DataFrame(pred4)
df = pd.read_csv("../input/titanic/gender_submission.csv")
data=pd.concat([df['PassengerId'],pred],axis=1)
data.columns=['PassengerId','Survived']
data.to_csv('sample_submission.csv',index=False)

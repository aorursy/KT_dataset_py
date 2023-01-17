# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import re
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
%matplotlib inline
sns.set()
import os
print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.info()
df_train.describe()
sns.countplot(x='Survived',data=df_train)#no of non survived persons are more
df_test['Survived']=0
df_test[['PassengerId','Survived']].to_csv('no_survivors.csv',index=False)#first model
sns.countplot(x='Sex',data=df_train)
sns.factorplot(x='Survived',col='Sex',kind='count',data=df_train)
df_train.groupby(['Sex']).Survived.sum()
print(df_train[df_train.Sex=='female'].Survived.sum()/df_train[df_train.Sex=='female'].Survived.count())
print(df_train[df_train.Sex=='male'].Survived.sum()/df_train[df_train.Sex=='male'].Survived.count())
df_test['Survived']=df_test.Sex=='female'
df_test['Survived']=df_test.Survived.apply(lambda x:int(x))
df_test[['PassengerId','Survived']].to_csv('women_survive.csv',index=False)#second model
survived_train=df_train.Survived
data=pd.concat([df_train.drop(['Survived'],axis=1),df_test.drop(['Survived'],axis=1)])
data.head()

data['Age']=data.Age.fillna(data.Age.mean())
data['Fare']=data.Fare.fillna(data.Fare.mean())
data['Embarked']=data.Embarked.fillna('S')
data['Embarked']=data.Embarked.map({'S':0,'C':1,'Q':2})
data['Sex']=data.Sex.map({'female':0,'male':1})
data['Has_cabin']=data.Cabin.apply(lambda x:0 if type(x)==float else 1)
data['Title']=data.Name.apply(lambda x:re.search('([A-Z][a-z]+)\.',x).group(1))
data['Title']=data.Title.replace({'Mlle':'Miss','Mme':'Mrs','Ms':'Miss','Master':'Mr'})
data['Title']=data.Title.replace(['Don','Dona','Rev','Dr','Major','Lady','Sir','Col','Capt','Countess','Jonkheer'],'Special')
data['Title']=data.Title.fillna(0)
data['Title']=data.Title.map({'Mr':0,'Mrs':1,'Miss':2,'Special':3})
data['CatAge']=pd.qcut(data.Age,q=4,labels=False)
data['CatFare']=pd.qcut(data.Fare,q=4,labels=False)
data['fam_size']=data.Parch+data.SibSp+1
data['IsAlone']=0
data.loc[data['fam_size']==1,'IsAlone']=1
data.drop(['Cabin','Name','PassengerId','Ticket','fam_size'],axis=1,inplace=True)
data=data.drop(['Age','Fare'],axis=1)
data=data.drop(['SibSp','Parch'],axis=1)
data.head(20)
data_dum=pd.get_dummies(data,drop_first=True)
data_train=data_dum.iloc[:891]
data_test=data_dum.iloc[891:]
X=data_train.values
test=data_test.values
y=survived_train.values
clf=KNeighborsClassifier()
k = np.arange(20)+1
parameters = {'n_neighbors': k}
#n_neighbors=[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
#algorithm=['auto']
#weights=['uniform','distance']
#leaf_size=list(range(1,50,5))
#param_grid={'algorithm':algorithm,'weights':weights,'leaf_size':leaf_size,'n_neighbors':k}
clf_cv=GridSearchCV(clf,parameters,cv=10)
clf_cv.fit(X,y)
y_pred=clf_cv.predict(test)
df_test['Survived']=y_pred
df_test[['PassengerId','Survived']].to_csv("KNN_model.csv",index=False)







# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#for data analysis
import random as rnd


#for data visulaisation
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#for machine learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
#Loading data
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
combine=[train,test]

train.columns
train.head()
ID=test['PassengerId']
train.info()
test.info()

#There are too many missing values in train['Cabin'],so we may drop that feature
train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)
train.describe()
train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values('Survived',ascending=False)
train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values('Survived',ascending=False)
train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values('Survived',ascending=False)
train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values('Survived',ascending=False)
#Now let us visualise how survival depends on quantitative variables

#Quatitative variables are Age,Price
g=sns.FacetGrid(train,col='Survived',size=4)
g.map(plt.hist,'Age',bins=20,alpha=0.5)
g=sns.FacetGrid(train,col='Survived',size=4)
g.map(plt.hist,'Fare',bins=20,alpha=0.5)
g=sns.FacetGrid(train,col='Survived',row='Pclass',size=4)
g.map(plt.hist,'Age',bins=20)
g=sns.FacetGrid(train,col='Survived',row='Pclass',size=4)
g.map(plt.hist,'Fare',bins=20)
g=sns.FacetGrid(train,col='Survived',row='Sex',size=4)
g.map(plt.hist,'Embarked')
#Data wrangling
#correcting by dropping features: Here we will drop the featire 'ticket number' as it is of not much use.
train.drop('Ticket',axis=1,inplace=True)
test.drop('Ticket',axis=1,inplace=True)
combine=[train,test]
train.columns
#creating new features
for dataset in combine:
    dataset['Title']=dataset.Name.str.extract('([A-Za-z]+\.)',expand=False)
pd.crosstab(train['Title'],train['Sex'])
train['Title'].value_counts()
train.head()
train['Title']=train['Title'].replace(['Dr','Rev','Col','Major','Lady','Capt','Sir','Jonkheer','Countess','Don'],'Rare',regex=True)
train['Title']=train['Title'].replace('Ms','Miss',regex=True)
train['Title']=train['Title'].replace('Mlle','Miss',regex=True)
train['Title']=train['Title'].replace('Mme','Mrs',regex=True)

test['Title']=test['Title'].replace(['Dr','Rev','Col','Major','Lady','Capt','Sir','Jonkheer','Countess','Don'],'Rare',regex=True)
test['Title']=test['Title'].replace('Ms','Miss',regex=True)
test['Title']=test['Title'].replace('Mlle','Miss',regex=True)
test['Title']=test['Title'].replace('Mme','Mrs',regex=True)

train['Title'].value_counts()
train.head(50)
train[['Title','Survived']].groupby('Title',as_index=False).mean()
combine=[train,test]
title_mapping = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Rare.": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()
train.head(10)
test.head()
train.drop(['PassengerId','Name'],axis=1,inplace=True)
test.drop(['PassengerId','Name'],axis=1,inplace=True)
train.head()
test.head()
gender_mapping={"female":1,"male":2}
for dt in combine:
   dt['Sex']= dt['Sex'].map(gender_mapping)
   dt['Sex']=dt['Sex'].fillna(0)
port_mapping={'S':1,'C':2,'Q':3}

for dt in combine:
    dt['Embarked']=dt['Embarked'].map(port_mapping)
    dt['Embarked']=dt['Embarked'].fillna(0)
train.head()
test.head()
#filling missing values
train.info()
grid=sns.FacetGrid(train,row='Pclass',col='Sex')
grid.map(plt.hist,'Age',alpha=0.5,bins=20)
grid.add_legend




age_guess=np.zeros((2,3))

for dt in combine:
    for i in range(1,3):
        for j in range(1,4):
            guess=dt[(dt['Sex']==i) & (dt['Pclass']==j)]['Age'].dropna()
            a=i-1
            b=j-1
            age_guess[a,b]=guess.median()
            
            
for dt in combine:
    for i in range(1,3):
        for j in range(1,4):
            a=i-1
            b=j-1
            dt.loc[(dt.Age.isnull())& (dt.Sex==i)&(dt.Pclass==j),'Age']=age_guess[a,b]
            
train.info()
train.head()
test.head()
train['Ageband']=pd.cut(train['Age'],5)
train[['Ageband','Survived']].groupby(['Ageband'],as_index=False).mean().sort_values(by='Ageband',ascending=True)
for dt in combine :
    dt.loc[dt['Age']<=16,'Age']=0
    dt.loc[(dt['Age']>16)&(dt['Age']<32),'Age']=1
    dt.loc[(dt['Age']>32)&(dt['Age']<48),'Age']=2
    dt.loc[(dt['Age']>48)&(dt['Age']<64),'Age']=3
    dt.loc[(dt['Age']>64),'Age']=4
train.head()    
train.drop('Ageband',axis=1,inplace=True)
train.head()
for dt in combine:
    dt['Familysize']=dt['SibSp']+dt['Parch']+1
    
train[['Familysize','Survived']].groupby('Familysize',as_index=False).mean().sort_values('Survived',ascending=False)
# we will check if how were the survival rates among the people whose Familysize is 1

for dt in combine:
    dt['Isalone']=0
    dt.loc[(dt['Familysize']==1),'Isalone']=1

train[['Isalone','Survived']].groupby('Isalone',as_index=False).mean()
#we will drop Sibso, Parch,familysize

for dt in combine:
    dt.drop(['SibSp','Parch','Familysize'],axis=1,inplace=True)
train.head()
train.info()
test.info()
test['Fare'].fillna(test['Fare'].dropna().median(),inplace=True)
test.info()
#create a fare band

train['fareband']=pd.qcut(train['Fare'],4)
train[['fareband','Survived']].groupby('fareband',as_index=False).mean().sort_values('Survived',ascending=True)
for dt in combine:
    dt.loc[(dt['Fare']<7.91),'Fare']=0
    dt.loc[(dt['Fare']>=7.91) & (dt['Fare']<14.45),'Fare']=1
    dt.loc[(dt['Fare']>=14.45) & (dt['Fare']<31),'Fare']=2
    dt.loc[(dt['Fare']>=31),'Fare']=3
train.drop('fareband',axis=1,inplace=True)
train.head()
test.head()
#model predict and solve

from sklearn.model_selection import train_test_split

Xtrain=train.drop('Survived',axis=1)
ytrain=train['Survived']

#Let us stack models to get better accuracy

#Our base models will depend on how correlated they are we will choose the ones that are uncorrelated.
#Let us define a class

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier)
from sklearn.svm import SVC


class base_predictors(object):
    def __init__(self,clf,params):
        params['random_state']=0
        self.clf=clf(**params)
    def train(self,X_train,y_train):
        self.clf.fit(X_train,y_train)
    def predict(self,X_test):
        return self.clf.predict(X_test)
    def fit(self,X_test,y_test):
        self.clf.fit(X_test,y_test)
    def feature_importances(self,X_test,y_test):
        return(self.clf.fit(X_test,y_test).feature_importances_)
    
    
from sklearn.model_selection import KFold
NFolds=5
ntrain=train.shape[0]
kfold = KFold(n_splits= NFolds, random_state=0)

#out of fold predictions

def out_of_fold(clf,X_train,y_train,X_test):
    oof_train=np.zeros(train.shape[0])
    oof_test=np.zeros(test.shape[0])
    oof_test_training_set=np.empty((NFolds,test.shape[0]))
    
    for i,(index_train,index_test) in enumerate(kfold.split(X_train)):
        x_tr=X_train[index_train]
        x_ts=X_train[index_test]
        y_tr=y_train[index_train]
        
        clf.train(x_tr,y_tr)
        oof_train[index_test]=clf.predict(x_ts)
        oof_test_training_set[i,:]=clf.predict(X_test)
        
    oof_test[:]=oof_test_training_set.mean(axis=0)
        
    return oof_test.reshape(-1,1),oof_train.reshape(-1,1)

#We will be using logistic regression ,random forests,svm,ada boost, gradient boosting

lr_params={}
rf_params={'n_estimators':100,'n_jobs':-1,'max_depth':3,'max_features':'sqrt'}
svc_params={'C':0.05,'kernel':'linear'}
ada_params={'n_estimators':100,'learning_rate':0.5}
grad_params={'n_estimators':100,'max_depth':3}

lr=base_predictors(clf=LogisticRegression,params=lr_params)
rf= base_predictors(clf=RandomForestClassifier,params=lr_params) 
svc=base_predictors(clf=SVC,params=svc_params)
ada=base_predictors(clf=AdaBoostClassifier,params=ada_params)
grad=base_predictors(clf=GradientBoostingClassifier,params=grad_params)




        
        
train.head()
y_train=train['Survived'].ravel()
X_train=train.drop('Survived',axis=1).values

X_test=test.values
X_train
    
lr_test,lr_train=out_of_fold(lr, X_train,y_train,X_test)
rf_test,rf_train=out_of_fold(rf,X_train,y_train,X_test)
svc_test,svc_train=out_of_fold(svc,X_train,y_train,X_test)
ada_test,ada_train=out_of_fold(ada,X_train,y_train,X_test)
grad_test,grad_train=out_of_fold(grad,X_train,y_train,X_test)
base_classifiers_predictions_train=np.concatenate((lr_train,rf_train,svc_train,ada_train,grad_train),axis=1)
base_classifiers_predictions_test=np.concatenate((lr_test,rf_test,svc_test,ada_test,grad_test),axis=1)

base_predictions=pd.DataFrame({'Logistic_Regression':lr_train.ravel(),
                              'Random_forests':rf_train.ravel(),
                              'svc':svc_train.ravel(),
                              'ada':ada_train.ravel(),
                              'grad':grad_train.ravel()}
                              )
base_predictions.head()
#let us look at the cprrelation between them
base_predictions.corr()
sns.heatmap(base_predictions.corr(),vmin=0,annot=True)

#We see that the the base models are higly correlated.Using models that are not highly correlated will give better results.


from xgboost import XGBClassifier

me_=XGBClassifier(learning_rate=0.05,n_estimators=1000)
me_.fit(base_classifiers_predictions_train,y_train)
predictions_=me_.predict(base_classifiers_predictions_test)
accuracy=round(me_.score(base_classifiers_predictions_train,y_train))
print(accuracy)
submissions=pd.DataFrame({'PassengerId':ID,'Survived':predictions_})
submissions.to_csv('Submissions.csv',index=False)



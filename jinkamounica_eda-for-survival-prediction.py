import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier)

from sklearn.svm import SVC



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/gender_submission.csv')
train.isnull().sum()
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True)

plt.title('Survived')

plt.ylabel('')
train.head()
train.columns

train.groupby(['Sex','Survived'])['Survived'].count()
sns.countplot('Sex',hue='Survived',data=train)

plt.title('Sex:Survived vs Dead')

plt.show()
train.groupby(['Pclass','Survived'])['Survived'].count()
sns.countplot('Pclass',hue='Survived',data=train)

plt.title('Sex:Survived vs Dead')

plt.show()
sns.catplot(x="Sex", y="Survived", col="Pclass",data=train, saturation=.5,

            kind="bar", ci=None, aspect=.6)
pd.crosstab([train.Sex,train.Survived],train.Pclass,margins=True).style.background_gradient(cmap='summer_r')
train['Age'].max(),train['Age'].min(),train['Age'].mean()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=train,split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
fig = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)

fig.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

fig.add_legend()
sns.factorplot('Embarked','Survived',data=train)

plt.show()
train[["SibSp", "Survived"]].groupby(['SibSp'],

        as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], 

            as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab(train.SibSp,train.Pclass).style.background_gradient(cmap='summer_r')
pd.crosstab(train.Parch,train.Pclass).style.background_gradient(cmap='summer_r')
train.isnull().sum()
sns.countplot('Embarked',data=train)

plt.title('Passengers embarked at each station')

plt.show()
train['Embarked'].fillna('S',inplace=True)
train['Embarked'].isnull().any()
train.head(5)
train['Salutation']=0

for i in train:

    train['Salutation']=train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
train
train['Salutation'].unique()
pd.crosstab(train.Sex,train.Salutation)
train['Salutation'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
train.groupby('Salutation')['Age'].mean()
train.loc[(train.Age.isnull())&(train.Salutation=='Mr'),'Age']=33

train.loc[(train.Age.isnull())&(train.Salutation=='Mrs'),'Age']=36

train.loc[(train.Age.isnull())&(train.Salutation=='Master'),'Age']=5

train.loc[(train.Age.isnull())&(train.Salutation=='Miss'),'Age']=22

train.loc[(train.Age.isnull())&(train.Salutation=='Other'),'Age']=46
train['Age'].isnull().sum()
sns.factorplot('Pclass','Survived',col='Salutation',data=train)

plt.show()
plt.hist(x = [train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()
train['Age_Range']=0

train.loc[train['Age']<=10,'Age_Range']=0

train.loc[(train['Age']>10)&(train['Age']<=20),'Age_Range']=1

train.loc[(train['Age']>20)&(train['Age']<=40),'Age_Range']=2

train.loc[(train['Age']>40)&(train['Age']<=50),'Age_Range']=3

train.loc[(train['Age']>50)&(train['Age']<=65),'Age_Range']=4

train.loc[train['Age']>65,'Age_Range']=5
sns.factorplot('Age_Range','Survived',data=train,col='Pclass')

plt.show()
train.drop(['PassengerId','Ticket','Cabin'],axis=1,inplace=True)
train['Fare'].nunique()
f,ax=plt.subplots(1,3,figsize=(20,8))

sns.distplot(train[train['Pclass']==1].Fare,ax=ax[0])

ax[0].set_title('Fares in Pclass 1')

sns.distplot(train[train['Pclass']==2].Fare,ax=ax[1])

ax[1].set_title('Fares in Pclass 2')

sns.distplot(train[train['Pclass']==3].Fare,ax=ax[2])

ax[2].set_title('Fares in Pclass 3')

plt.show()

plt.hist(x = [train[train['Survived']==1]['Fare'], train[train['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()
train['Fare_Range']=0

train.loc[train['Fare']<=7.91,'Fare_Range']=0

train.loc[(train['Fare']>7.91)&(train['Fare']<=14.454),'Fare_Range']=1

train.loc[(train['Fare']>14.454)&(train['Fare']<=31),'Fare_Range']=2

train.loc[(train['Fare']>31)&(train['Fare']<=513),'Fare_Range']=3
sns.heatmap(train.corr(),annot=True,cmap='summer_r',linewidths=0.2) 

plt.show()
train['FamilySize']=train['SibSp']+train['Parch']+1

train['IsAlone'] = 1 

train['IsAlone'].loc[train['FamilySize'] > 1] = 0
train.columns
train.drop(['Name','Age','Fare'],axis=1,inplace=True)
train.columns
train.head()
train['Sex'].replace(['male','female'],[0,1],inplace=True)

train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

train['Salutation'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
type(train)
train.head()
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})

fig=plt.gcf()

fig.set_size_inches(15,15)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
from sklearn.linear_model import LogisticRegression 

from sklearn import svm 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier 

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xg



from sklearn.model_selection import train_test_split 

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score
train_f,test_f=train_test_split(train,test_size=0.3,random_state=0,stratify=train['Survived'])

train_X=train_f.drop(['Survived'],axis=1)

train_Y=train_f['Survived']

test_X=test_f.drop(['Survived'],axis=1)

test_Y=test_f['Survived']

X=train.drop(['Survived'],axis=1)

Y=train['Survived']
lrc = LogisticRegression()

lrc.fit(train_X,train_Y)

prediction_lrc=lrc.predict(test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction_lrc,test_Y))
svc=svm.SVC(kernel='rbf',C=1,gamma=0.1)

svc.fit(train_X,train_Y)

prediction_svc=svc.predict(test_X)

print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction_svc,test_Y))
dtc=DecisionTreeClassifier()

dtc.fit(train_X,train_Y)

prediction_dtc=dtc.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction_dtc,test_Y))
knn=KNeighborsClassifier() 

knn.fit(train_X,train_Y)

prediction_knn=knn.predict(test_X)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction_knn,test_Y))
rfc=RandomForestClassifier(n_estimators=80)

rfc.fit(train_X,train_Y)

prediction_rfc=model.predict(test_X)

print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction_rfc,test_Y))
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)

result=cross_val_score(ada,X,Y,cv=10,scoring='accuracy')

print('The cross validated score for AdaBoost is:',result.mean())
ada.fit(train_X,train_Y)
sub = pd.DataFrame()

sub['PassengerId'] = test['PassengerId']

sub['Survived'] = ada.predict_proba(test)

sub.to_csv("sub.csv",index=False)
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)

result=cross_val_score(grad,X,Y,cv=10,scoring='accuracy')

print('The cross validated score for Gradient Boosting is:',result.mean())
xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)

result=cross_val_score(xgboost,X,Y,cv=10,scoring='accuracy')

print('The cross validated score for XGBoost is:',result.mean())
f,ax=plt.subplots(2,2,figsize=(15,12))

model=RandomForestClassifier(n_estimators=500,random_state=0)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])

ax[0,0].set_title('Feature Importance in Random Forests')

model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1])

ax[0,1].set_title('Feature Importance in AdaBoost')

model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0])

ax[1,0].set_title('Feature Importance in Gradient Boosting')

model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1])

ax[1,1].set_title('Feature Importance in XgBoost')

plt.show()
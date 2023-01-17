import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')

train.info()

test.info()
train.describe()
train.head()
train.isnull().sum()
train.isnull().count()
total=train.isnull().sum()

missing_data_perc=(total/train.isnull().count())*100

mis_data=round(missing_data_perc.sort_values(ascending=False),1)

missing_data=pd.concat([total,mis_data],axis=1,keys=['Total','%'])

missing_data
train.columns.values
survived='survived'

not_survived='not survived'

fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(9,4))

women=train[train['Sex']=='female']

men=train[train['Sex']=='male']

ax=sns.distplot(women[women['Survived']==1].Age.dropna(),label=survived,ax=axes[0],kde=False)

ax=sns.distplot(women[women['Survived']==0].Age.dropna(),label=not_survived,ax=axes[0],kde=False)

ax.legend()

ax.set_title('Female')

ax=sns.distplot(men[men['Survived']==1].Age.dropna(),label=survived,ax=axes[1],kde=False)

ax=sns.distplot(men[men['Survived']==0].Age.dropna(),label=not_survived,ax=axes[1],kde=False)

ax.legend()

ax.set_title('Male')

sns.barplot(x='Pclass',y='Survived',data=train)

print('Percentage of Pclass=1 who survived is',train['Survived'][train['Pclass']== 1].value_counts(normalize=True)[1]*100) 

print('Percentage of Pclass=2 who survived is',train['Survived'][train['Pclass']== 2].value_counts(normalize=True)[1]*100) 

print('Percentage of Pclass=3 who survived is',train['Survived'][train['Pclass']== 3].value_counts(normalize=True)[1]*100) 

sns.barplot(x='Sex',y='Survived',data=train)

print('Percentage of female who survived is',train['Survived'][train['Sex']=='female'].value_counts(normalize=True)[1]*100) 

print('Percentage of male who survived is',train['Survived'][train['Sex']=='male'].value_counts(normalize=True)[1]*100)                        



sns.barplot(x='SibSp',y='Survived',data=train)

print('Percentage of Sibsp=0 who survived is',train['Survived'][train['SibSp']==0].value_counts(normalize=True)[1]*100) 

print('Percentage of Sibsp=1 who survived is',train['Survived'][train['SibSp']==1].value_counts(normalize=True)[1]*100) 

print('Percentage of Sibsp=2 who survived is',train['Survived'][train['SibSp']==2].value_counts(normalize=True)[1]*100) 

print('Percentage of Sibsp=3 who survived is',train['Survived'][train['SibSp']==3].value_counts(normalize=True)[1]*100) 

print('Percentage of Sibsp=4 who survived is',train['Survived'][train['SibSp']==4].value_counts(normalize=True)[1]*100) 

sns.barplot(x='Parch',y='Survived',data=train)

print('Percentage of Parch=0 who survived is',train['Survived'][train['Parch']==0].value_counts(normalize=True)[1]*100) 

print('Percentage of Parch=1 who survived is',train['Survived'][train['Parch']==1].value_counts(normalize=True)[1]*100) 

print('Percentage of Parch=2 who survived is',train['Survived'][train['Parch']==2].value_counts(normalize=True)[1]*100) 

print('Percentage of Parch=3 who survived is',train['Survived'][train['Parch']==3].value_counts(normalize=True)[1]*100) 



train['Age']=train['Age'].fillna(-0.5)

test['Age']=test['Age'].fillna(-0.5)

bins=[-1,0,5,12,17,25,35,60,np.inf]

labels=['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior Citizen']

train['Agegroup']=pd.cut(train['Age'],bins,labels=labels)

test['Agegroup']=pd.cut(train['Age'],bins,labels=labels)



sns.barplot(x='Agegroup',y='Survived',data=train)
train['Cabin_b']=train['Cabin'].notnull().astype('int')

test['Cabin_b']=test['Cabin'].notnull().astype('int')



print('Percentage of people of Cabin = 1 who survived is',train['Survived'][train['Cabin_b']==1].value_counts(normalize=True)[1]*100)

print('Percentage of people of Cabin = 0 who survived is',train['Survived'][train['Cabin_b']==0].value_counts(normalize=True)[1]*100)

sns.barplot(x='Cabin_b',y='Survived',data=train)
train.columns

test.columns
train=train.drop(['Cabin'],axis=1)

test=test.drop(['Cabin'],axis=1)
train=train.drop(['Ticket'],axis=1)

test=test.drop(['Ticket'],axis=1)
train=train.drop(['Fare'],axis=1)

test=test.drop(['Fare'],axis=1)
print('No. of people embarked from S :',train[train['Embarked']=='S'].shape[0])

print('No. of people embarked from C :',train[train['Embarked']=='C'].shape[0])

print('No. of people embarked from Q :',train[train['Embarked']=='Q'].shape[0])
train['Embarked']=train['Embarked'].fillna('S')
test['Embarked']=test['Embarked'].fillna('S')
train.isnull().sum()
test.isnull().sum()
train.info()
train=train.drop(['Name'],axis=1)

test=test.drop(['Name'],axis=1)
train.info()
test.info()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
train['Sex']=le.fit_transform(train['Sex'])

train['Embarked']=le.fit_transform(train['Embarked'])

train['Agegroup']=le.fit_transform(train['Agegroup'].cat.codes)
test['Sex']=le.fit_transform(test['Sex'])

test['Embarked']=le.fit_transform(test['Embarked'])

test['Agegroup']=le.fit_transform(test['Agegroup'].cat.codes)
train.info()
test.info()
train.head()
test.head()
trainn=train.drop(['Age'],axis=1)

testn=test.drop(['Age'],axis=1)
trainn.info()
testn.info()
X_train=trainn.drop(['PassengerId','Survived'],axis=1)

y_train=trainn['Survived']

X_test=testn.drop(['PassengerId'],axis=1)



X_train.shape
y_train.shape
X_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

LR=LogisticRegression()

LR.fit(X_train,y_train)

y_pred=LR.predict(X_test)

acc_logi=accuracy_score(y_train,LR.predict(X_train))*100

print('Accuracy is',acc_logi)

y_pred.shape
from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(criterion='entropy',min_samples_split=100)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

acc_dec=accuracy_score(y_train,clf.predict(X_train))*100

print('Accuracy is',acc_dec)

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100)

rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

acc_ran=accuracy_score(y_train,rf.predict(X_train))*100

print('Accuracy is',acc_ran)

from sklearn.svm import SVC

sv=SVC()

sv.fit(X_train,y_train)

y_pred=sv.predict(X_test)

acc_svm=accuracy_score(y_train,sv.predict(X_train))*100

print('Accuracy is',acc_svm)
from sklearn.neighbors import KNeighborsClassifier

knc=KNeighborsClassifier(n_neighbors=4,weights='uniform',algorithm='auto',leaf_size=30)

knc.fit(X_train,y_train)

y_pred=knc.predict(X_test)

acc_knn=accuracy_score(y_train,knc.predict(X_train))*100

print('Accuracy is',acc_knn)
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(X_train,y_train)

y_pred=nb.predict(X_test)

acc_nb=accuracy_score(y_train,nb.predict(X_train))*100

print('Accuracy is',acc_nb)
accu_score=pd.DataFrame({'Model':['Logistic Regression','Decision Tree','Random Forest','KNN','Support Vector Machine',

                        'Naive Bayes'],

                         'Accuracy' :[acc_logi,acc_dec,acc_ran,acc_knn,acc_svm,acc_nb]

                          })

accu_score=accu_score.sort_values(by='Accuracy',ascending=False)

print('Accuracies are :',accu_score)
submission=pd.DataFrame({'PassengerId' :test['PassengerId'],

                         'Survived' : y_pred    

})
#submission.to_csv('../input/submission/submission.csv',index=False)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

% matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')

train_df.head()
train_df.info()

print('------------')

test_df.info()
train_df.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)

test_df.drop(['Ticket','Name'],axis=1,inplace=True)
train_df.groupby(['Embarked']).count()
#Fill colums Embarked the missing values with S as most occured value

train_df['Embarked'].fillna('S', inplace=True)
# plot

sns.factorplot('Embarked', 'Survived', data=train_df, size=4, aspect=3,ci=None)

#fig, (axis1,axis2,axis3)=plt.subplots(1,2, figsize=(10,5))

#sns.factorplot('Embarked', data=train_df, kind='count', ax=axis1)

#sns.factorplot('Survived',hue='Embarked',data=train_df,kind='count',ax=axis2)

sns.factorplot('Survived', col='Embarked', data=train_df,kind='count',size=5.0,aspect=0.6)
train_df.Sex.isnull().sum()
train_df.Age.isnull().sum()
sns.factorplot(x='Age',hue='Sex',row='Pclass',kind='count',data=train_df[train_df.Age.notnull()],size=5, aspect=1.2)
sns.factorplot(x='Sex',y='Survived', col='Pclass',data=train_df, kind='bar',ci=None,aspect=0.6)

sns.factorplot(x='Embarked',y='Survived',data=train_df, kind='bar',ci=None,aspect=1.5)
fig,(axis1,axis2,axis3)=plt.subplots(1,3, figsize=(15,5))

sns.countplot(x='Embarked',data=train_df,ax=axis1)

sns.countplot(x='Survived', hue='Embarked',data=train_df,ax=axis2)

gr=train_df[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean()

gr

sns.barplot(x='Embarked', y='Survived',data=gr,order=['S','C','Q'],ci=None,ax=axis3)
embark_dummies_train=pd.get_dummies(train_df['Embarked'])

embark_dummies_train.drop('S', axis=1,inplace=True)



embark_dummies_test=pd.get_dummies(test_df['Embarked'])

embark_dummies_test.drop('S', axis=1,inplace=True)



train_df=train_df.join(embark_dummies_train)

test_df=test_df.join(embark_dummies_test)



train_df.drop('Embarked', axis=1, inplace=True)

test_df.drop('Embarked',axis=1, inplace=True)
train_df.head()
#Fare

test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

test_df['Fare'].dtypes

test_df['Fare']=test_df['Fare'].astype(int)

train_df['Fare']=train_df['Fare'].astype(int)
fare_not_survived = train_df["Fare"][train_df["Survived"] == 0]

fare_survived=train_df['Fare'][train_df['Survived']==1]
average_fare=pd.DataFrame([fare_not_survived.mean(),fare_survived.mean()])

std_fare=pd.DataFrame([fare_not_survived.std(),fare_survived.std()])

train_df['Fare'].plot(kind='hist',figsize=(15,3),bins=100,xlim=(0,50))

average_fare.index.names=std_fare.index.names=['Survived']

average_fare.plot(yerr=std_fare, kind='bar')

std_fare.plot(kind='bar')
# Age



fig,(axis1,axis2)=plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age values - Train')

axis2.set_title('New Age values - Train')



average_age_train=train_df['Age'].mean()

std_age_train=train_df['Age'].std()

count_nan_age_train=train_df['Age'].isnull().sum()



average_age_test=test_df['Age'].mean()

std_age_test=test_df['Age'].std()

count_nan_age_test=test_df['Age'].isnull().sum()



rand_1 = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train, size = count_nan_age_train)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)



train_df['Age'].dropna().astype(int).hist(bins=70,ax=axis1)



train_df.loc[train_df['Age'].isnull(), 'Age']=rand_1

#train_df['Age'].loc[train_df['Age'].isnull()]=rand_1

print(train_df.isnull().sum())

test_df.loc[test_df['Age'].isnull(),'Age']=rand_2

print(test_df.isnull().sum())



train_df['Age']=train_df['Age'].astype(int)

test_df['Age']=test_df['Age'].astype(int)



train_df['Age'].hist(bins=70, ax=axis2)



                           



                          
#peaks for survived/not survived passengers by their age

facet=sns.FacetGrid(train_df, hue='Survived', aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,train_df['Age'].max()))

facet.add_legend()

#average survived passengers by age

fig,axis1=plt.subplots(1,1,figsize=(18,4))

average_age=train_df[['Age','Survived']].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age',y='Survived', data=average_age)

train_df['Family']=train_df['Parch']+train_df['SibSp']

train_df['Family']=np.where(train_df['Family']>0,1,0)



test_df['Family']=test_df['Parch']+test_df['SibSp']

test_df['Family']=np.where(test_df['Family']>0,1,0)



train_df.drop(['Parch','SibSp'],axis=1, inplace=True)

test_df.drop(['Parch','SibSp'],axis=1, inplace=True)



fig,(axis1,axis2)=plt.subplots(1,2, sharex=True, figsize=(10,5))

sns.countplot(x='Family',data=train_df,ax=axis1)

fp=train_df[['Family','Survived']].groupby(['Family'],as_index=False).mean()

sns.barplot(x='Family',y='Survived',data=fp,ax=axis2)



axis1.set_xticklabels(['With Family','Alone'],rotation=0)
#Sex

train_df['Person']=np.where(train_df['Age']<16,'child',train_df['Sex'])

test_df['Person']=np.where(test_df['Age']<16,'child',test_df['Sex'])

train_df.drop('Sex', axis=1, inplace=True)

test_df.drop('Sex', axis=1, inplace=True)



person_dummies_train=pd.get_dummies(train_df['Person'])

person_dummies_test=pd.get_dummies(test_df['Person'])



person_dummies_train.head()





person_dummies_train.drop('male', axis=1,inplace=True)

person_dummies_test.drop('male', axis=1,inplace=True)
train_df=train_df.join(person_dummies_train)

test_df=test_df.join(person_dummies_train)
fig,(axis1,axis2)=plt.subplots(1,2,figsize=(15,5))

sns.countplot(x='Person', data=train_df,ax=axis1)

pm=train_df[['Person','Survived']].groupby(['Person'], as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=pm, ax=axis2)
train_df.drop('Person', axis=1,inplace=True)

test_df.drop('Person', axis=1,inplace=True)

train_df.drop('Cabin', axis=1,inplace=True)

test_df.drop('Cabin', axis=1,inplace=True)
#sns.factorplot('Pclass','Survived', data=train_df, size=5)

Pclass_dummies_train=pd.get_dummies(train_df['Pclass'], prefix='Class')

Pclass_dummies_test=pd.get_dummies(test_df['Pclass'], prefix='Class')



Pclass_dummies_train.drop('Class_3', axis=1, inplace=True)

Pclass_dummies_test.drop('Class_3', axis=1, inplace=True)



train_df.drop('Pclass',axis=1, inplace=True)

test_df.drop('Pclass',axis=1, inplace=True)

train_df.head()
X=train_df.drop('Survived',axis=1)

y=train_df['Survived']

X_test=test_df.drop('PassengerId', axis=1)
train_df.head()
#linear Regression

from sklearn.linear_model import LinearRegression

LiR=LinearRegression()

LiR.fit(X,y)

Y_pred_lir=LiR.predict(X_test)

LiR.score(X,y)
# Logistic Regression

LR=LogisticRegression()

LR.fit(X,y)

Y_pred_lr=LR.predict(X_test)

LR.score(X,y)
# SVM

svc=SVC()

svc.fit(X,y)

Y_pred_svm=svc.predict(X_test)

svc.score(X,y)
# Random　Forests

RF=RandomForestClassifier(n_estimators=100)

RF.fit(X,y)

Y_pred_rf=RF.predict(X_test)

RF.score(X,y)
#knn

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X,y)

Y_pred_knn=knn.predict(X_test)

knn.score(X,y)
#Gausian Naive Bayes

NB=GaussianNB()

NB.fit(X,y)

Y_pred_nb=NB.predict(X_test)

NB.score(X,y)

coef=pd.DataFrame(train_df.columns.delete(0))

coef.columns=['Features']

coef['Coefficient Estimate']=pd.Series(LR.coef_[0])

coef
Submission=pd.DataFrame({'PassengerID':test_df['PassengerId'],

                        'Survived':Y_pred_rf

                        })

Submission.to_csv('titanic.csv', index=False)
Y_pred_lr
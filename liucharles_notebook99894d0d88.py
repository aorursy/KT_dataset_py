# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame,Series

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')

#preview the data



train_df.head()
train_df.info()

print('--------------------------------')

test_df.info()
#drop the unimportant columns

train_df=train_df.drop(['PassengerId','Name','Ticket'],axis=1)

test_df=test_df.drop(['Name','Ticket'],axis=1)
#Embarked

#show the Embarked

set(train_df['Embarked'])

print('S={}'.format(np.sum(train_df['Embarked']=='S')))

print('Q={}'.format(np.sum(train_df['Embarked']=='Q')))      

print('C={}'.format(np.sum(train_df['Embarked']=='C')))

print('Nan={}'.format(np.sum(train_df['Embarked'].isnull())))
#fill the null data with 'S'

train_df['Embarked']=train_df['Embarked'].fillna('S')
#plot

sns.factorplot('Embarked','Survived',data=train_df,size=4,aspect=3)



fig,(axis1,axis2,axis3)=plt.subplots(1,3,figsize=(15,5))



sns.countplot(x='Embarked',data=train_df,ax=axis1)

sns.countplot(x='Survived',hue='Embarked',data=train_df,order=[0,1],ax=axis2)

embark_perc=train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked',y='Survived',data=embark_perc,order=['S','Q','C'],ax=axis3)
embark_dummies_train  = pd.get_dummies(train_df['Embarked'])

embark_dummies_train.drop(['S'],axis=1,inplace=True)



embark_dummies_test  = pd.get_dummies(test_df['Embarked'])

embark_dummies_test.drop(['S'],axis=1,inplace=True)



train_df=train_df.join(embark_dummies_train)

test_df=test_df.join(embark_dummies_test)



train_df.drop(['Embarked'],axis=1,inplace=True)

test_df.drop(['Embarked'],axis=1,inplace=True)
#Fare



#only the test has missing data

test_df['Fare'].fillna(test_df['Fare'].median(),inplace=True)



train_df['Fare']=train_df['Fare'].astype(int)

test_df['Fare']=test_df['Fare'].astype(int)



fare_not_survived=train_df['Fare'][train_df['Survived']==0]

fare_survived=train_df['Fare'][train_df['Survived']==1]



avgerge_fare=DataFrame([fare_not_survived.mean(),fare_survived.mean()])

std_fare=DataFrame([fare_not_survived.std(),fare_survived.std()])



train_df['Fare'].plot(kind='hist',figsize=(15,3),bins=100,xlim=(0,50))



avgerge_fare.index.names=std_fare.index.names=['Survived']

avgerge_fare.plot(yerr=std_fare,kind='bar',legend=False)
#Age



fig,(axis1,axis2)=plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age Values---Titanic')

axis2.set_title('New Age Values---Titanic')



avgerge_age_train=train_df['Age'].mean()

std_age_train=train_df['Age'].std()

count_nan_age_train=np.sum(train_df['Age'].isnull())



avgerge_age_test=test_df['Age'].mean()

std_age_test=test_df['Age'].std()

count_nan_age_test=np.sum(test_df['Age'].isnull())



rand_train=np.random.randint(avgerge_age_train-std_age_train,avgerge_age_train+std_age_train,size=count_nan_age_train)

rand_test=np.random.randint(avgerge_age_test-std_age_test,avgerge_age_test+std_age_test,size=count_nan_age_test)



train_df['Age'].dropna().astype(int).hist(bins=70,ax=axis1)

train_df['Age'][np.isnan(train_df['Age'])]=rand_train

test_df['Age'][np.isnan(test_df['Age'])]=rand_test



train_df['Age']=train_df['Age'].astype(int)

test_df['Age']=test_df['Age'].astype(int)



train_df['Age'].hist(bins=70,ax=axis2)
facet=sns.FacetGrid(train_df,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,train_df['Age'].max()))

facet.add_legend()



fig,axis1=plt.subplots(1,1,figsize=(18,4))

avgerge_age=train_df[['Age','Survived']].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age',y='Survived',data=avgerge_age,ax=axis1)
#Cabin

#Most of Cabin is null

train_df.drop('Cabin',axis=1,inplace=True)

test_df.drop('Cabin',axis=1,inplace=True)
#Family



train_df['Family']=train_df['Parch']+train_df['SibSp']

train_df['Family'].loc[train_df['Family']>0]=1

train_df['Family'].loc[train_df['Family']==0]=0



test_df['Family']=test_df['Parch']+test_df['SibSp']

test_df['Family'].loc[test_df['Family']>0]=1

test_df['Family'].loc[test_df['Family']==0]=0



train_df=train_df.drop(['Parch','SibSp'],axis=1)

test_df=test_df.drop(['Parch','SibSp'],axis=1)



fig,(axis1,axis2)=plt.subplots(1,2,sharex=True,figsize=(10,5))



sns.countplot(x='Family',data=train_df,order=[1,0],ax=axis1)



family_perc=train_df[['Family','Survived']].groupby(['Family'],as_index=False).mean()



sns.barplot(x='Family',y='Survived',data=family_perc,order=[1,0],ax=axis2)



axis1.set_xticklabels(['With Family','Alone'],rotation=0)
# Sex

def get_person(passenger):

    age,sex=passenger

    return 'child' if age<16 else sex



train_df['Person']=train_df[['Age','Sex']].apply(get_person,axis=1)

test_df['Person']=test_df[['Age','Sex']].apply(get_person,axis=1)



train_df.drop(['Sex'],axis=1,inplace=True)

test_df.drop(['Sex'],axis=1,inplace=True)



person_dummies_train  = pd.get_dummies(train_df['Person'])

person_dummies_train.columns = ['Child','Female','Male']

person_dummies_train.drop(['Male'], axis=1, inplace=True)



person_dummies_test=pd.get_dummies(test_df['Person'])

person_dummies_test.columns=['Child','Female','Male']

person_dummies_test.drop(['Male'],axis=1,inplace=True)



train_df=train_df.join(person_dummies_train)

test_df=test_df.join(person_dummies_test)



fig,(axis1,axis2)=plt.subplots(1,2,figsize=(10,5))



sns.countplot(x='Person',data=train_df,ax=axis1)



person_perc=train_df[['Person','Survived']].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person',y='Survived',data=person_perc,order=['male','female','child'],ax=axis2)



train_df.drop(['Person'],axis=1,inplace=True)

test_df.drop(['Person'],axis=1,inplace=True)



# Pclass

sns.factorplot('Pclass','Survived',order=[1,2,3],data=train_df,size=5)



pclass_dummies_train=pd.get_dummies(train_df['Pclass'])

pclass_dummies_train.columns=['Class_1','Class_2','Class_3']

pclass_dummies_train.drop(['Class_3'],axis=1,inplace=True)



pclass_dummies_test=pd.get_dummies(test_df['Pclass'])

pclass_dummies_test.columns=['Class_1','Class_2','Class_3']

pclass_dummies_test.drop(['Class_3'],axis=1,inplace=True)



train_df.drop(['Pclass'],axis=1,inplace=True)

test_df.drop(['Pclass'],axis=1,inplace=True)



train_df=train_df.join(pclass_dummies_train)

test_df=test_df.join(pclass_dummies_test)
X_train=train_df.drop(['Survived'],axis=1)

Y_train=train_df['Survived']

X_test=test_df.drop(['PassengerId'],axis=1).copy()
# Logistic Regression

logreg=LogisticRegression()

logreg.fit(X_train,Y_train)

Y_pred=logreg.predict(X_test)

logreg.score(X_train,Y_train)
# Random Forests

random_forest=RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train,Y_train)

random_forest.predict(X_test)

random_forest.score(X_train,Y_train)
coeff_df=DataFrame(train_df.columns.delete(0))

coeff_df.columns=['Features']

coeff_df['Coefficient Estimation']=pd.Series(logreg.coef_[0])

coeff_df
submission=DataFrame({'PassengerId':test_df['PassengerId'],'Survived':Y_pred})

submission.to_csv('titanic_pre.csv',index=False)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')
train.head()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

#identifying the missing values
sns.set_style('whitegrid')

sns.countplot(train['Survived'], hue= train['Pclass'])
sns.distplot(train['Age'].dropna(), bins=30)
sns.countplot(x='SibSp', data=train)
train['Fare'].hist( bins=40, figsize=(8,4))
import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind='hist',bins=50)
plt.figure(figsize=(10,7))

sns.boxplot(x='Pclass',y='Age', data=train)
def impute_age(cols):

    Age=cols[0]

    Pclass=cols [1]

    

    if pd.isnull(Age):

        

        if Pclass==1:

            return 38

        elif Pclass==2:

            return 29

        else:

            return 25

    else:

        return Age
train['Age']= train[['Age', 'Pclass']].apply(impute_age, axis=1)
plt.figure(figsize=(10,7))

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
train.drop('Cabin', axis=1, inplace=True)
train.head()
Sex= pd.get_dummies(train['Sex'],drop_first=True)

Embarked= pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)

train.drop(['PassengerId'], axis=1, inplace=True)
train= pd.concat((train,Sex,Embarked), axis=1)
train.head()
x= train.drop(['Survived'] , axis=1)

y= train['Survived']
from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(x, y)
test=pd.read_csv('../input/test.csv')

test.info()

test.head()
Sex= pd.get_dummies(test['Sex'],drop_first=True)

Embarked= pd.get_dummies(test['Embarked'],drop_first=True)

test.drop(['Name', 'Sex', 'Ticket', 'Embarked','Cabin'], axis=1, inplace=True)

test= pd.concat((test,Sex,Embarked), axis=1)
test.head()
sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.boxplot(x=test['Pclass'],y=test['Fare'])
def impute_age(cols):

    Age=cols[0]

    Pclass=cols [1]

    

    if pd.isnull(Age):

        

        if Pclass==1:

            return 40.91

        elif Pclass==2:

            return 28.77

        else:

            return 24.02

    else:

        return Age
test['Age']= test[['Age', 'Pclass']].apply(impute_age, axis=1)
test[test['Pclass']==3].describe()
test[test.isnull().any(axis=1)]
test[test['PassengerId']==1044]
means = test.groupby('Pclass')['Fare'].transform('mean')

test['Fare'] = test['Fare'].fillna(means)
predict=logmodel.predict(test.drop(['PassengerId'],axis=1))
submission=pd.read_csv('../input/gender_submission.csv')

submission['Survived']=predict

submission.to_csv('titanic_submission.csv',index=False)
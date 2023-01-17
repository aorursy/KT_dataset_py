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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train=pd.read_csv('../input/train.csv')
train.head()
train.info()
train.describe()
sns.heatmap(train.isnull(),cbar=False)
sns.set_style('darkgrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='Spectral')
sns.distplot(train['Age'].dropna(),kde=False,bins=40)
sns.countplot(x='SibSp',data=train)
sns.boxplot(x='Pclass',y='Age',data=train)
m_1=train[train['Pclass']==1]['Age'].median()

m_2=train[train['Pclass']==2]['Age'].median()

m_3=train[train['Pclass']==3]['Age'].median()
def imputeage(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass ==1:

            return m_1

        elif Pclass ==2:

            return m_2

        else:

            return m_3

        

    else:

        return Age
train=train.drop(['Cabin','Name','Ticket','PassengerId'],axis=1)

train['Age']=train[['Age','Pclass']].apply(imputeage,axis=1)
sns.heatmap(train.isnull(),cbar=False)
sex=pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked'],axis=1,inplace=True)
train=pd.concat([train,sex,embark],axis=1)
train.head()
from sklearn.linear_model import LogisticRegression
test=pd.read_csv('../input/test.csv')

test.head()
test=test.drop(['Cabin','Name','Ticket','PassengerId'],axis=1)

test['Age']=test[['Age','Pclass']].apply(imputeage,axis=1)
sex=pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked'],axis=1,inplace=True)
test=pd.concat([test,sex,embark],axis=1)
sns.heatmap(test.isnull(),cbar=False)
fm_1=train[train['Pclass']==1]['Fare'].median()

fm_2=train[train['Pclass']==2]['Fare'].median()

fm_3=train[train['Pclass']==3]['Fare'].median()
def imputefare(cols):

    Fare=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Fare):

        if Pclass ==1:

            return fm_1

        elif Pclass ==2:

            return fm_2

        else:

            return fm_3

        

    else:

        return Fare
test['Fare']=train[['Fare','Pclass']].apply(imputefare,axis=1)
sns.heatmap(test.isnull(),cbar=False)
test.head()
train.head()
X_train=train.drop('Survived',axis=1)

y_train=train['Survived']

X_test=test
logmodel=LogisticRegression()

logmodel.fit(X_train,y_train)
pred=logmodel.predict(X_test)
len(pred)
predictions=pd.read_csv('../input/test.csv')
predictions['Survived']=pred
predictions.head()
sns.heatmap(predictions.isnull(),cbar=False)
predictions_formated=predictions.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
predictions_formated.head()
predictions_formated.to_csv('My_formated_titanic_predictions', index=False)
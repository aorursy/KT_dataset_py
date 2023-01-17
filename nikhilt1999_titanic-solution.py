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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

train = pd.read_csv('../input/train.csv')

test =  pd.read_csv('../input/test.csv')

print(train.head())

test.shape

test.head()


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.countplot(x='SibSp',data=train)

train['Fare'].hist(color='green',bins=40,figsize=(8,4))
import cufflinks as cf

cf.go_offline()
train['Fare'].iplot(kind='hist',bins=30,color='green')
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
train.head()
train.dropna(inplace=True)


sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.shape
train = pd.concat([train,sex,embark],axis=1)
X_test.shape
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)

test.drop('Cabin',axis=1,inplace=True)

test.dropna(inplace=True)

sex = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)

test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test = pd.concat([test,sex,embark],axis=1)
from sklearn.model_selection import train_test_split
train.head()
X_train,X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.3, 

                                                    random_state=101)
X_train.shape
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(test.head())

test.shape
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age

        

        

test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)















test.shape
predictions=logmodel.predict(test)
predictions.shape
#checking accuracy using test data

from sklearn.metrics import accuracy_score , confusion_matrix

acc_logistic_regression  =accuracy_score(y_test , predictions)

print(acc_logistic_regression)

print("Confusion matrix\n",confusion_matrix(y_test , predictions))
X_test.head(5)

X_test.info()
X_train.head(5)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
logmodel.score(X_train,y_train)
submission = pd.read_csv("../input/gender_submission.csv")



print(submission.shape)

submission=pd.DataFrame({"PassengerId":test.PassengerId,"Survived":predictions})





submission.to_csv("submission1.csv",index=False)
X_train.shape
y_train.shape
X_test.shape
predictions.shape
submission.shape
submission
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

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=test,palette='winter')
#Replace age of Pclass 1, Pclass 2, Pclass 3 with 37, 29, 24 respectively in train data

#Replace age of Pclass 1, Pclass 2, Pclass 3 with 42, 26, 24 respectively in test data

def impute_age_train(cols):

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

    

    

    

def impute_age_test(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 42



        elif Pclass == 2:

            return 26



        else:

            return 24



    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age_train,axis=1)

test['Age'] = test[['Age','Pclass']].apply(impute_age_test,axis=1)
#Lets drop 'Cabin' column and row with some missing values

train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)



train.dropna(inplace=True)

test.fillna(test.mean(),inplace=True)
train.info()
test.info()
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train = pd.concat([train,sex,embark],axis=1)

train.head()
sex = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)

test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test = pd.concat([test,sex,embark],axis=1)

test.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.30, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

pred_logmodel = logmodel.predict(X_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=25)

knn.fit(X_train,y_train)

pred_knn = knn.predict(X_test)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

pred_dtree = dtree.predict(X_test)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=300)

rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
#Logistic Regression

print(classification_report(y_test,pred_logmodel))

print(confusion_matrix(y_test,pred_logmodel))
#KNN Model

print(classification_report(y_test,pred_knn))

print(confusion_matrix(y_test,pred_knn))
#Decision Tree Classifier

print(classification_report(y_test,pred_dtree))

print(confusion_matrix(y_test,pred_dtree))
#Random Forest Classifier

print(classification_report(y_test,pred_rfc))

print(confusion_matrix(y_test,pred_rfc))
#Since best accuracy is coming from Random Forest Classifier(84%), we would take that.

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=300)

rfc.fit(train.drop('Survived',axis=1), train['Survived'])

pred_rfc = rfc.predict(test) 
submission= pd.DataFrame({

    "PassengerId" : test["PassengerId"],

    'Survived' : pred_rfc

})



submission.to_csv('submission.csv',index=False)
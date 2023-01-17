# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='RdBu_r')
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

test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)

sex_test = pd.get_dummies(test['Sex'],drop_first=True)

embark_test = pd.get_dummies(test['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket','Fare'],axis=1,inplace=True)

test.drop(['Sex','Embarked','Name','Ticket','Fare'],axis=1,inplace=True)

train = pd.concat([train,sex,embark],axis=1)

test = pd.concat([test,sex_test,embark_test], axis = 1)
test.shape
# train.dropna(inplace=True)

#test.dropna(inplace=True)

test.columns[test.isnull().any()].tolist()
X_train = train.drop("Survived", axis=1)

y_train = train["Survived"]

X_test  = test
from sklearn.ensemble import RandomForestClassifier
logmodel = RandomForestClassifier()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
X_train.shape, y_train.shape, X_test.shape
acc_log = round(logmodel.score(X_train, y_train) * 100, 2)

acc_log
print(predictions)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })
submission.to_csv('submission.csv', index=False)
submission.shape
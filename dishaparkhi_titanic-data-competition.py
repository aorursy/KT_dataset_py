# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()
sns.heatmap(train.isnull(),yticklabels = False,cbar=False,cmap = 'viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue = 'Sex',data = train,palette ='RdBu_r')
sns.countplot(x='Survived',hue = 'Pclass',data = train)
sns.distplot (train['Age'].dropna(),kde=False,bins = 30)
train.info()
sns.countplot(x='SibSp',data = train)
train['Fare'].hist(bins=40,figsize=(10,4))
plt.figure(figsize=(10,7))

sns.boxplot(x='Pclass',y='Age',data = train)
def impute_age(cols):

    Age = cols[0]

    Pclass=cols[1]

    

    

    if pd.isnull(Age):

        

        if Pclass== 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(), yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin', axis=1, inplace=True)
sns.heatmap(train.isnull(), yticklabels=False,cbar=False,cmap='viridis')
train.dropna(inplace=True)
sns.heatmap(train.isnull(), yticklabels=False,cbar=False,cmap='viridis')
pd.get_dummies(train['Sex'])
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()
train.drop(['Sex','Embarked','Name','Ticket'],axis=1, inplace=True)
train.head()
#train.drop(['PassengerId'],axis=1, inplace=True)
train.head()
X = train.drop('Survived',axis=1)

y=train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.30, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report 
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.head()
sns.heatmap(test.isnull(), yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(10,7))

sns.boxplot(x='Pclass',y='Age',data = test)
def impute_age(cols):

    Age = cols[0]

    Pclass=cols[1]

    

    

    if pd.isnull(Age):

        

        if Pclass== 1:

            return 42

        elif Pclass == 2:

            return 26

        else:

            return 24

    else:

        return Age
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(test.isnull(), yticklabels=False,cbar=False,cmap='viridis')
test.drop('Cabin', axis=1, inplace=True)
sns.heatmap(test.isnull(), yticklabels=False,cbar=False,cmap='viridis')
test.dropna(inplace=True)
sns.heatmap(test.isnull(), yticklabels=False,cbar=False,cmap='viridis')
test.head()
pd.get_dummies(test['Sex'])
sex = pd.get_dummies(test['Sex'],drop_first=True)
embark =pd.get_dummies(test['Embarked'],drop_first=True)
test = pd.concat([test,sex,embark],axis=1)
test.head()
test.drop(['Sex','Embarked','Name','Ticket'],axis=1, inplace=True)
#test.drop(['PassengerId'],axis=1, inplace=True)
test.head()
X_test = test

#y_test=test['Survived']
predictions = logmodel.predict(X_test)
submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = predictions

submission.head()
submission.to_csv('my_submission.csv', index = False)
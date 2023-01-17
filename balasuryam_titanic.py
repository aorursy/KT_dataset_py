# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/titanic/train.csv')
train.head(3)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
sns.countplot(train['Survived'])
sns.countplot(train['Survived'] ,hue = train['Pclass'] )
sns.distplot(train['Age'].dropna(), kde= False)
train.info()
sns.countplot(train['SibSp'])
sns.countplot(train['Parch'])
sns.distplot(train['Fare'], kde = False, bins = 50)
import cufflinks as cf

cf.go_offline()
#train['Fare'].iplot(kind = 'hist')
sns.boxplot(train['Pclass'], train['Age'])
def impute_age(cols):

    age = cols[0]

    Pclass = cols[1]

    if pd.isnull(age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 30

        else:

            return 27

    return age
train['Age'] = train[['Age','Pclass']].apply(impute_age ,axis=1)
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
train.drop('Cabin',inplace= True, axis = 1)
train.head(3)
train.info()
train.dropna(inplace = True)
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
Sex = pd.get_dummies(train['Sex'], drop_first=True)
Embarked = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train,Sex,Embarked], axis = 1)
train.head(3)
train.drop(['Sex','Name','Ticket','Embarked'],axis=1,inplace = True)
train.head(3)
train.tail(3)
train.drop(['PassengerId'],axis=1,inplace = True)
#Leaving Pclass as 1,2,3

train.head(3)
import xgboost

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

classifier = xgboost.XGBClassifier(learning_rate=0.03)

X = train.drop('Survived',axis = 1)

y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
test = pd.read_csv('../input/titanic/test.csv')

test.head(3)

test['Age'] = test[['Age','Pclass']].apply(impute_age ,axis=1)

Sex = pd.get_dummies(test['Sex'], drop_first=True)

Embarked = pd.get_dummies(test['Embarked'], drop_first=True)

test = pd.concat([test,Sex,Embarked],axis = 1)

#test.head(3)

x = test.drop(['PassengerId','Name','Sex','Cabin','Ticket','Embarked'], axis=1)

x['Fare'] = x['Fare'].fillna(x['Fare'].mean())

predictions = classifier.predict(x)

final = pd.DataFrame(test['PassengerId'])

final['Survived'] = predictions

final.info()

final.head(3)

final.to_csv('submission.csv')
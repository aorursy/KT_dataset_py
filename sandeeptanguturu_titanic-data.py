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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train=pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
test=pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
train.info()
train.describe()
train.isnull().sum()
train[train['Age'].isnull()]
train[train['Age'].isnull()]['Pclass'].value_counts()
pd.crosstab(train['Pclass'],train['Age'])
train[train['Pclass']==3]['Age'].mean()
train[train['Pclass']==2]['Age'].mean()
train[train['Pclass']==1]['Age'].mean()
for i in train[train['Age'].isnull()]['Pclass']==1:
    train['Age'].fillna(value=38.0,inplace=True)
for i in train[train['Age'].isnull()]['Pclass']==2:
    train['Age'].fillna(value=30.0,inplace=True)
for i in train[train['Age'].isnull()]['Pclass']==3:
    train['Age'].fillna(value=25.0,inplace=True)
train[train['Age'].isnull()]['Pclass'].value_counts()
train.isnull().sum()
train[train['Embarked'].isnull()]
train[train['Pclass']==1]['Embarked'].value_counts()
train['Embarked'].fillna(value='S',inplace=True)
train.isnull().sum()/len(train)
train.drop('Cabin',axis=1,inplace=True)
train.head(2)
train.drop(['Name','Ticket','PassengerId'],axis=1,inplace=True)
train.head(2)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train['Sex']=le.fit_transform(train['Sex'])
train['Embarked']=le.fit_transform(train['Embarked'])
train.info()
plt.figure(figsize=(15,8))
sns.heatmap(train.corr(),annot=True,linewidths=0.8)
plt.show()
X=train.drop('Survived',axis=1)
y=train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=10000)
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,accuracy_score
print("The Training Score of Logistic Regression is :",lr.score(X_train,y_train))
print("The Testing Score of Logistic Regression is :",lr.score(X_test,y_test))
print("The Accuracy of Logistic Regression is :",accuracy_score(y_test,pred))
print('\n')
print("The Confusion Matrix of Logistic Regression is : \n ",confusion_matrix(y_test,pred))
print('\n')
print("The Classification Report of Logistic Regression is : \n ",classification_report(y_test,pred))
print('\n')
print("The F1-Score of Logistic Regression is :",f1_score(y_test,pred))
test.info()
test.describe()
test.isnull().sum()
test[test['Age'].isnull()]
test[test['Age'].isnull()]['Pclass'].value_counts()
pd.crosstab(test['Pclass'],test['Age'])
test[test['Pclass']==1]['Age'].mean()
test[test['Pclass']==2]['Age'].mean()
test[test['Pclass']==3]['Age'].mean()
for i in test[test['Age'].isnull()]['Pclass']==1:
    test['Age'].fillna(value=41.0,inplace=True)
for i in test[test['Age'].isnull()]['Pclass']==2:
    test['Age'].fillna(value=29.0,inplace=True)
for i in test[test['Age'].isnull()]['Pclass']==3:
    test['Age'].fillna(value=24.0,inplace=True)
test.isnull().sum()
test[test['Fare'].isnull()]
test[test['Pclass']==3]['Fare'].mean()
test['Fare'].fillna(value=12.45,inplace=True)
test.isnull().sum()/len(test)
test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
test.head(2)
test['Sex']=le.fit_transform(test['Sex'])
test['Embarked']=le.fit_transform(test['Embarked'])
c=pd.DataFrame()
c['PassengerId']=test['PassengerId']
test.drop('PassengerId',axis=1,inplace=True)
plt.figure(figsize=(15,8))
sns.heatmap(test.corr(),annot=True,linewidths=0.8)
plt.show()
X_test1=test
pred_test=lr.predict(X_test1)
dataframe=pd.DataFrame()
dataframe['Survived']=pd.Series(pred_test)
final=pd.concat((c,dataframe),axis=1)
final.head()

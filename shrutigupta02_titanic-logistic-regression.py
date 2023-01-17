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
#for preprocessing
import pandas as pd
import numpy as np

#For visualization
import matplotlib.pyplot as plt
import seaborn as sns

#For modelling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#for accuracy metric
from sklearn.metrics import accuracy_score 
#for confusion matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
train=pd.read_csv("/kaggle/input/titanic/train.csv")
test=pd.read_csv("/kaggle/input/titanic/test.csv")
train.head()
print("Train Shape: ",train.shape)
print("Test Shape: ", test.shape)
train.info()
train.describe()
train.isnull().sum()
train=train.drop('Cabin',axis=1)
train['Age'] = train['Age'].fillna(train['Age'].median())

train.isnull().sum()
train.info()
train.dropna(inplace=True)
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train = pd.concat([train,sex,embark],axis=1)
train.drop(['Sex','Embarked','Name','Ticket','PassengerId','Fare'],axis=1,inplace=True)
train.head()
test.shape
test.info()
test.head()
test.isnull().sum()
test=test.drop('Cabin',axis=1)
train.dropna(inplace=True)
test['Age'] = test['Age'].fillna(test['Age'].median())
test.isnull().sum()
sex = pd.get_dummies(test['Sex'],drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)
test = pd.concat([test,sex,embark],axis=1)
passenger_id=test["PassengerId"]
test.drop(['Sex','Embarked','Name','Ticket','PassengerId','Fare'],axis=1,inplace=True)
test.isnull().sum()
test.head()
sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',data=train,hue='Pclass')
sns.countplot(x='SibSp',data=train)
sns.distplot(train['Age'],bins=30,kde=False)
X = train.drop('Survived',axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.head()
lg_model = LogisticRegression()
lg_model.fit(X_train,y_train)
lg_model.score(X_train,y_train)
lg_model.score(X_test,y_test)
y_pred=lg_model.predict(X_test)
pd.DataFrame(y_pred).head()
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test,y_pred)
#visualize confusion matrix 

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('cm', y=1.05, size=15)
cm
y_test_pred=lg_model.predict(test)
submission = pd.DataFrame({
        "PassengerId": passenger_id,
        "Survived": y_test_pred
    })

submission.to_csv('./submission.csv', index=False)
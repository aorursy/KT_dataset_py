import numpy as np #Analysis
import pandas as pd #Analysis
import seaborn as sns #Visualization
import matplotlib
import matplotlib.pyplot as plt #Visualization
%matplotlib inline
matplotlib.style.use('ggplot')
sns.set_style('white')

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(5)
train.info()
train.describe()
train.describe(include=['O'])
sns.countplot(x='Survived',hue='Pclass',data=train)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.countplot(x='Survived',hue='Sex',data=train)
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
grid = sns.FacetGrid(train,row='Embarked',col='Survived')
grid.map(sns.barplot,'Sex','Fare',ci=None)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False)
train.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
train.drop(['Fare'],axis=1,inplace=True)
test.drop(['Fare'],axis=1,inplace=True)
train.head()
test.head()
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0}).astype(int)
test['Sex'] = test['Sex'].map({'female': 1, 'male': 0}).astype(int)
train.head(1)
test.head(1)
train['Age'].fillna(value=train['Age'].median(),inplace=True)
test['Age'].fillna(value=train['Age'].median(),inplace=True)
train['Embarked'].fillna(value=train['Embarked'].mode()[0],inplace=True)
test['Embarked'].fillna(value=train['Embarked'].mode()[0],inplace=True)
train['Embarked'] = train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)
test['Embarked'] = test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)
print(train.info())
print(test.info())
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
x_all = train.drop(['Survived', 'PassengerId'], axis=1)
y_all = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=100)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_prediction = rfc.predict(X_test)
rfc_score=accuracy_score(y_test, rfc_prediction)
print(rfc_score)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg_prediction = logreg.predict(X_test)
logreg_score=accuracy_score(y_test, logreg_prediction)
print(logreg_score)
X_train = train.drop(['Survived', 'PassengerId'], axis=1)
X_test = test.drop(['PassengerId'], axis=1)
y_train = train['Survived']
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_prediction = rfc.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": rfc_prediction
    })
submission.to_csv('Titanic_RFC.csv', index=False)

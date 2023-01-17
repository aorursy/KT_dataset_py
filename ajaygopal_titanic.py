

import pandas as pd

import numpy as np



# For Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# To ignore warnings:

import warnings

warnings.filterwarnings('ignore')



# To display all the rows and columns:

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)


gend_submission=pd.read_csv("../input/titanic/gender_submission.csv")
train_df = pd.read_csv("../input/titanic/train.csv")
test_df=pd.read_csv("../input/titanic/test.csv")
gend_submission.head()
test_df.head()
train_df.head()
train_df.drop(['Name'],axis=1,inplace=True)

test_df.drop(['Name'],axis=1,inplace=True)

train_df.drop(['Ticket'],axis=1,inplace=True)

test_df.drop(['Ticket'],axis=1,inplace=True)
round(100*(train_df.isnull().sum()/len(train_df.index)),2)
train_df.fillna(train_df['Age'].mean(),inplace=True)
train_df.drop(['Cabin'],axis=1,inplace=True)
train_df= train_df.fillna({"Embarked": "S"})
round(100*(test_df.isnull().sum()/len(test_df.index)),2)
test_df.drop(['Cabin'],axis=1,inplace=True)
test_df.fillna(test_df['Age'].mean(),inplace=True)
train_df.fillna(train_df['Fare'].mean(),inplace=True)
sns.countplot(x='Pclass',hue='Survived',data=train_df)
sns.countplot(x='Sex',hue='Survived',data=train_df)
sns.countplot(x='SibSp',hue='Survived',data=train_df)
sns.countplot(x='Parch',hue='Survived',data=train_df)
sns.countplot(x='Embarked',hue='Survived',data=train_df)
sns.boxplot(y = "Fare", x = "Survived", data = train_df)
dummy1 = pd.get_dummies(train_df[['Sex','Embarked']], drop_first=True)

dummy1.head()
train_df = pd.concat([train_df, dummy1], axis=1)

train_df.head()
train_df=train_df.drop(['Sex','Embarked'], axis = 1)
dummy2 = pd.get_dummies(test_df[['Sex','Embarked']], drop_first=True)

dummy2.head()
test_df = pd.concat([test_df, dummy2], axis=1)

test_df.head()
test_df=test_df.drop(['Sex','Embarked'], axis = 1)
test_df.head()
train_df.head()
X_train=train_df.drop(['PassengerId','Survived'],axis=1)
y_train=train_df['Survived']
X_test=test_df.drop(['PassengerId'],axis=1)
X_train.shape
X_test.shape
X_train.drop(['Embarked_C'],axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler



# storing column names in cols,

cols = X_train.columns

X_train = pd.DataFrame(StandardScaler().fit_transform(X_train))

X_train.columns = cols

X_train.columns
# storing column names in cols,

cs = X_test.columns

X_test = pd.DataFrame(StandardScaler().fit_transform(X_test))

X_test.columns = cs

X_test.columns
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
#import xgboost as xgb

#from xgboost import XGBClassifier

#xgboost = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
#y_pred = xgboost.predict(X_test)
#from sklearn.tree import DecisionTreeClassifier

#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(n_estimators=100)

#rf.fit(X_train, y_train)

#y_pred = rf.predict(X_test)
#dt = DecisionTreeClassifier()

#dt.fit(X_train, y_train)

#y_pred = dt.predict(X_test)
my_submission = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': y_pred})

my_submission.to_csv('g_submission.csv', index=False)
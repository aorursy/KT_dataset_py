import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/titanic/train.csv')

df.head(20)
df.info()
df.info()
df.shape
df.isnull().sum()
df_1 = df.drop(columns = {'Name','Cabin'})

df_1.head(20)
df_1.isnull().sum()
df_1['Age'] = df_1['Age'].fillna(df_1['Age'].mean())
df_1.dropna(inplace = True)
df_1.isnull().sum()
df_1.shape
df_1.info()
print(df_1['Sex'].nunique())

print(df_1['Ticket'].nunique())

print(df_1['Embarked'].nunique())
df_2 = pd.get_dummies(df_1[['Sex','Embarked']])   #The get_dummies() function is used to convert categorical variable into dummy/indicator variables.

df_2.head(20)
df_2.tail(20)
df_1 = df_1.join(df_2)

df_1.head(20)
df_1.drop(columns={'Sex' , 'Embarked' , 'Ticket'}, axis=1 , inplace=True)

df_1.info()
plt.figure(figsize=(16,8), dpi= 80)

sns.heatmap(df_1.corr(), cmap='RdYlGn', center=0)

plt.title('Correlation Matrix (for Titanic Dataset)')
y = df_1['Survived']

X = df_1.drop(columns={'Survived'}, axis=1)

X.head(20)
df_test = pd.read_csv('../input/titanic/test.csv')

df_test.info()
df_test.drop(columns = {'Name','Cabin','Ticket'},axis = 1, inplace = True)

df_test.info()
df_test_2 = pd.get_dummies(df_test[['Sex','Embarked']])

df_test_2.head(20)
df_test = df_test.join(df_test_2)

df_test.head(20)
df_test.drop(columns = {'Sex','Embarked'}, axis =1 ,inplace =True)

df_test.head(20)
df_test.info()
df_test.isnull().sum()
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())

df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median()) 
df_test.isnull().sum()
X.drop(columns = {'PassengerId'}, inplace = True, axis =1)

X.head(20)
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X, y)

print("Accuracy:{:.2f} %".format(classifier.score(X, y)*100))
!pip install catboost
from catboost import CatBoostClassifier

classifier_1 = CatBoostClassifier()

classifier_1.fit(X, y)

print("Accuracy:{:.2f} %".format(classifier_1.score(X, y)*100))
sub = pd.DataFrame(columns =['PassengerId','Survived'])

sub['PassengerId'] = df_test['PassengerId'].astype(int)

sub.head()
df_test.drop(columns = {"PassengerId"}, inplace =True, axis = 1)

y_pred = classifier.predict(df_test)
sub['Survived'] = y_pred.astype(int)
sub.head(20)
sub.to_csv('Submission.csv', index=False)
col_sorted_by_importance=classifier.feature_importances_.argsort()

feat_imp=pd.DataFrame({

    'cols':X.columns[col_sorted_by_importance],

    'imps':classifier.feature_importances_[col_sorted_by_importance]

})



#!pip install plotly-express

import plotly_express as px

px.bar(feat_imp, x='cols', y='imps')
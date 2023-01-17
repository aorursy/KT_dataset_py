import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

sns.set()

%matplotlib inline
raw_data= pd.read_csv('../input/titanic/train.csv')

raw_data.head()
raw_data.shape
percent = (raw_data.isnull().sum()/raw_data.isnull().count()).sort_values(ascending=False)

percent
raw_data=raw_data.drop('Cabin' , axis=1)

raw_data=raw_data.drop('PassengerId' , axis=1)

raw_data=raw_data.drop('Name' , axis=1)

raw_data=raw_data.drop('Ticket' , axis=1)



raw_data.head()
sns.heatmap(raw_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(12,6))

sns.boxplot(x='Pclass', y="Age", data=raw_data)
plt.figure(figsize=(12,6))

sns.boxplot(x='Sex', y="Age", data=raw_data)
raw_data['Age'] = raw_data.groupby(['Pclass','Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
sns.heatmap(raw_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
raw_data.dropna(axis=0,inplace=True)

sns.heatmap(raw_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(x='Sex',data=raw_data)

raw_data.shape
sns.countplot(x='Pclass',data=raw_data)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=raw_data)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=raw_data)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Embarked',data=raw_data)
sns.set_style('whitegrid')

sns.violinplot(x='Survived', y='Fare', data=raw_data)
corrmat = raw_data.corr()

plt.figure(figsize=(12, 9))

sns.heatmap(corrmat, annot=True, square=True)
raw_data.info()
train=pd.get_dummies(raw_data,drop_first=True)
train.head()
test=pd.read_csv('../input/titanic/test.csv')
test.head()
test.isnull().sum()
test=test.drop('Cabin' , axis=1)

test=test.drop('PassengerId' , axis=1)

test=test.drop('Name' , axis=1)

test=test.drop('Ticket' , axis=1)

test=pd.get_dummies(test,drop_first=True)
test['Age'] = test.groupby(['Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
test['Fare'].describe()
test['Fare'].fillna(14,inplace=True)
test['Fare'].fillna(14,inplace=True)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test=pd.get_dummies(test,drop_first=True)
test.head()
X_train=train.drop('Survived',axis=1)

Y_train=train['Survived']
from sklearn.ensemble import RandomForestClassifier

Model=RandomForestClassifier()

Model.fit(X_train,Y_train)
y_pred = Model.predict(test)
y_pred
predict=pd.DataFrame(y_pred)

sub_df=pd.read_csv('../input/titanic/gender_submission.csv')
dataset=pd.concat([sub_df['PassengerId'],predict],axis=1)

dataset.columns=['PassengerId','Survived']
dataset.to_csv(r'D:\Linear Regression\gender_submission.csv',index=False)
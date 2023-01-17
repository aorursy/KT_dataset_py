# Load Libraries

import pandas as pd

from pandas import Series,DataFrame

import csv

import sklearn

from sklearn.linear_model import LogisticRegression
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
df=pd.read_csv("../input/train.csv")
df.head()
df.tail()
df.info()
df.shape
df.describe()
df['Age'].fillna(df['Age'].median(), inplace=True)
df.describe()
sns.barplot(x="Sex", y="Survived", data=df)
survived_sex=df[df['Survived']==1]['Sex'].value_counts()

dead_sex=df[df['Survived']==0]['Sex'].value_counts()

df_survived=pd.DataFrame([survived_sex, dead_sex])

df_survived.index=['Survived','Dead']

df_survived.plot(kind='bar',stacked=True)
plt.hist([df[df['Survived']==1]['Age'], df[df['Survived']==0]['Age']], stacked=True, color=['g','r'], bins=30,

         label=['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend()
plt.hist([df[df['Survived']==1]['Fare'],df[df['Survived']==0]['Fare']], stacked=True, color=['g','r'],bins=30, label=['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend()
ax = plt.subplot()

ax.set_ylabel('Average fare')

df.groupby('Pclass').mean()['Fare'].plot(kind='bar',ax = ax)
sns.factorplot('Sex', kind='count', data=df)
sns.factorplot('Pclass',kind='count',data=df, hue='Sex')
xt=pd.crosstab(df['Pclass'],df['Survived'])

xt
xt.plot(kind='bar',stacked=True, title='Survival Rate by Passenger Classes')

plt.xlabel('Passenger Class')

plt.ylabel('Survival Rate')
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import preprocessing
np.random.seed(12)
label_encoder=preprocessing.LabelEncoder()
# Convert sex and embarked variables to numeric

df['Sex']=label_encoder.fit_transform(df['Sex'].astype('str'))

df['Embarked']=label_encoder.fit_transform(df['Embarked'].astype('str'))
# Initialize the model

rf_model=RandomForestClassifier(n_estimators=1000, max_features=2,oob_score=True)

features=['Sex','Pclass','Embarked','Age','Fare']
# Train the model

rf_model.fit(X=df[features],y=df['Survived'].astype('str'))

print("OOB accuracy: ")

print(rf_model.oob_score_)
for feature, imp in zip(features,rf_model.feature_importances_):

    print(feature,imp)
test=pd.read_csv("../input/test.csv")
test.describe()
test['Age'].fillna(test['Age'].median(), inplace=True)
test.describe()
# Convert sex and embarked variables of test dataset to numeric

test['Sex']=label_encoder.fit_transform(test['Sex'].astype('str'))

test['Embarked']=label_encoder.fit_transform(test['Embarked'].astype('str'))
test.head()
test.fillna(test.mean(), inplace=True)
# Predictions for test set

test_preds = rf_model.predict(X=test[features])
submission=pd.DataFrame({"PassengerId": test["PassengerId"], "Survived":test_preds})

submission.to_csv('titanic1.csv', index=False)
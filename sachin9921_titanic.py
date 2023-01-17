import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head()
test.head()
train.info()
print('='*40)
test.info()
train.describe().T

# relation btw Pclass and Survived
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# relation btw Sex & Servived
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# realtion btw SibSp and Survived
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# realtion btw Parch & Survived
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train['Survived'].value_counts(normalize=True)
g = sns.countplot(y=train['Survived'])
sns.barplot(x="Sex", y="Survived", data=train)
plt.show()
fig, axarr = plt.subplots(1, 2, figsize=(12,6))
a = sns.countplot(train['Sex'], ax=axarr[0]).set_title('Passengers count by sex')
axarr[1].set_title('Survival rate by sex')
b = sns.barplot(x='Sex', y='Survived', data=train, ax=axarr[1]).set_ylabel('Survival rate')
fig, axarr = plt.subplots(1,2,figsize=(12,6))
a = sns.countplot(x='Pclass', hue='Survived', data=train, ax=axarr[0]).set_title('Survivors and deads count by class')
axarr[1].set_title('Survival rate by class')
b = sns.barplot(x='Pclass', y='Survived', data=train, ax=axarr[1]).set_ylabel('Survival rate')
plt.title('Survival rate by sex and class')
g = sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train).set_ylabel('Survival rate')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train.Survived)
plt.title('Number of passenger Survived');

plt.subplot(1,2,2)
sns.countplot(x="Survived", hue="Sex", data=train)
plt.title('Number of passenger Survived');
plt.figure(figsize=(15,5))
plt.style.use('fivethirtyeight')

plt.subplot(1,2,1)
sns.countplot(train['Pclass'])
plt.title('Count Plot for PClass');

plt.subplot(1,2,2)
sns.countplot(x="Survived", hue="Pclass", data=train)
plt.title('Number of passenger Survived');
plt.title('Survival rate by sex and class')
g = sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train).set_ylabel('Survival rate')
train['Age'].plot(kind='hist')
train['Age'].hist(bins=40)
plt.title('Age Distribution')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train['Embarked'])
plt.title('Number of Port of embarkation')

plt.subplot(1,2,2)
sns.countplot(x="Survived", hue="Embarked", data=train)
plt.legend(loc='right')
plt.title('Number of passenger Survived')
# heatmap to check the corr.
sns.heatmap(train.corr(), annot=True)
train.isnull().sum()
test.isnull().sum()
# Replace missing age values with median age calculated per class

train["Age"].fillna(train["Age"].median(), inplace = True)
#Same for test set
test["Age"].fillna(test["Age"].median(), inplace = True)
# only for train (fill na value with 'S')
train["Embarked"].fillna("S", inplace = True)
# only for test ( update the missing value with median value)
test["Fare"].fillna(test["Fare"].median(), inplace = True)
# replace the missing value with 'unon'
train['Cabin'] = train['Cabin'].fillna('Unon')
test['Cabin'] = test['Cabin'].fillna('Unon')
train.isnull().sum()
test.isnull().sum()
sns.pairplot(train)
# Sex is categorical data so we can replace male to 0 and female to 1
# Same for Embarked

train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2
# We can combine SibSp and Parch into one synthetic feature called family size, 
# which indicates the total number of family members on board for each member.

train["FamSize"] = train["SibSp"] + train["Parch"] + 1
test["FamSize"] = test["SibSp"] + test["Parch"] + 1
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score 
features = ["Pclass", "Sex", "Age", "Embarked", "Fare", "FamSize"]
x = train[features] #define training features set
y = train["Survived"] #define training label set
X_test = test[features] #define testing features set
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 1)
x_train.shape
x_test.shape
reg_mod = LogisticRegression()
reg_mod.fit(x_train, y_train)
reg_mod.score(x_train, y_train)
pred = reg_mod.predict(x_test)
acc = accuracy_score(y_test, pred)
print(acc)
X_test.head()


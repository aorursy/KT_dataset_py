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
# data analysis and wrangling

import numpy as np

import numpy as np

import pandas as pd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set()   # set visualisation style

import matplotlib.style as style

style.use('fivethirtyeight')
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

import xgboost

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.metrics import accuracy_score
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
display(train.head())

display(test.head())

display(submission.head())
train_original = train.copy()

test_original = test.copy()
display(train.shape)

display(test.shape)
# does not include NaN values

display(train.count())

display(test.count())
display(train.dtypes)

display(test.dtypes)
numeric_cols_train = train.select_dtypes(exclude=object)

display(numeric_cols_train.head())

print('\n')

numeric_cols_train.columns
numeric_cols_train.shape
Categorical_cols_train = train.select_dtypes(include=object)

display(Categorical_cols_train.head())

print('\n')

Categorical_cols_train.columns
Categorical_cols_train.shape
numeric_cols_test = test.select_dtypes(exclude=object)

display(numeric_cols_test.head())

print('\n')

numeric_cols_test.columns
numeric_cols_test.shape
Categorical_cols_test = test.select_dtypes(include=object)

display(Categorical_cols_test.head())

print('\n')

Categorical_cols_test.columns
Categorical_cols_test.shape
display(train.isnull().sum().sort_values(ascending=False))

display(test.isnull().sum().sort_values(ascending=False))
missing_train = train.isnull().sum().sort_values(ascending=False)

missing_train = missing_train[missing_train>0]

missing_train
total = missing_train

percent = round(missing_train/len(train)*100, 2)[round(missing_train/len(train)*100, 2) != 0]

pd.concat([total, percent], axis=1, keys=['Total','Percent'])
plt.figure(figsize=(8,7))

missing_train.plot.bar()
plt.figure(figsize=(17,10))

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
missing_train[missing_train>223].sort_values(ascending=False)
missing_test = test.isnull().sum().sort_values(ascending=False)

missing_test = missing_test[missing_test>0]

missing_test
total = missing_test

percent = round(missing_test/len(test)*100, 2)[round(missing_test/len(test)*100, 2) != 0]

pd.concat([total, percent], axis=1, keys=['Total','Percent'])
plt.figure(figsize=(8,7))

missing_test.plot.bar()
plt.figure(figsize=(17,10))

sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='viridis')
missing_test[missing_test>105].sort_values(ascending=False)
plt.figure(figsize=(13,9))

sns.heatmap(numeric_cols_train.corr(), annot=True, cbar=False, cmap='viridis', linewidth=1, fmt='.1f', square=True)
plt.figure(figsize=(13,9))

corr = numeric_cols_train.corr()

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.3)], annot=True, cbar=False,

                                 cmap='viridis', linewidth=1, fmt='.1f', square=True)
# Numerical features

train.describe()
# for categorical features

train.describe(include=['O'])
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab(train['Pclass'],train['Survived'], values=train['Survived'], aggfunc='mean').style.background_gradient(cmap='summer_r')
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train['Survived'].value_counts()
f,ax = plt.subplots(1,2,figsize=(18,8))

train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=train, ax=ax[1])

ax[1].set_title('Survived')

plt.show()
sns.boxplot(train.Pclass)
sns.boxplot(train.Age)
sns.boxplot(train.Fare)
pd.crosstab(train['Sex'], train['Survived'], margins=True, margins_name="Total").style.background_gradient(cmap='summer_r')
print(train[train.Sex == 'female'].Survived.sum()/train[train.Sex == 'female'].Survived.count())

print(train[train.Sex == 'male'].Survived.sum()/train[train.Sex == 'male'].Survived.count())
print(train['Sex'].value_counts())

sns.countplot(x='Sex', data=train)
train.groupby(['Sex','Survived'])['Survived'].count()
f,ax=plt.subplots(1,2,figsize=(18,8))

train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived',data=train, ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
sns.catplot(x='Survived', col='Sex', kind='count', data=train)
pd.crosstab(train['Pclass'], train['Survived'], margins=True, margins_name="Total").style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))

train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number Of Passengers By Pclass')

ax[0].set_ylabel('Count')

sns.countplot('Pclass', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('Pclass:Survived vs Dead')

plt.show()
pd.crosstab([train['Sex'], train['Survived']], train['Pclass'], margins=True, margins_name="Total").style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass','Survived', hue='Sex', data=train)

plt.show()
sns.catplot(x='Survived', col='Pclass', kind='count', data=train)
display(train[train['Embarked'].isnull()])

display(train['Embarked'].isnull().sum())
display(test[test['Embarked'].isnull()])

display(test['Embarked'].isnull().sum())
# Filling the missing values in Embarked with S

train['Embarked'] = train['Embarked'].fillna('S')

#test['Embarked'] = test['Embarked'].fillna('S')
pd.crosstab([train['Embarked'], train['Pclass']],[train['Sex'], train['Survived']], margins=True, margins_name="Total").style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(2,2,figsize=(20,15))

sns.countplot('Embarked',data=train,ax=ax[0,0])

ax[0,0].set_title('No. Of Passengers Boarded')

sns.countplot('Embarked',hue='Sex',data=train,ax=ax[0,1])

ax[0,1].set_title('Male-Female Split for Embarked')

sns.countplot('Embarked',hue='Survived',data=train,ax=ax[1,0])

ax[1,0].set_title('Embarked vs Survived')

sns.countplot('Embarked',hue='Pclass',data=train,ax=ax[1,1])

ax[1,1].set_title('Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2,hspace=0.5)

plt.show()
sns.catplot(x='Survived', col='Embarked', kind='count', data=train)
train_corr = train.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

train_corr[train_corr['Feature 1'] == 'Age']
# Filling the missing values in Age with the medians of Sex and Pclass groups

train['Age'] = train.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

test['Age'] = test.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
print('Oldest Passenger was of:',train['Age'].max(),'Years')

print('Youngest Passenger was of:',train['Age'].min(),'Years')

print('Average Age on the ship:',train['Age'].mean(),'Years')
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=train, split=True, ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=train, split=True, ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
pd.crosstab([train['SibSp']], train['Survived'], margins=True, margins_name="Total").style.background_gradient(cmap='summer_r')
sns.barplot('SibSp','Survived',data=train)
pd.crosstab(train['SibSp'],train['Pclass'], margins=True, margins_name="Total").style.background_gradient(cmap='summer_r')
pd.crosstab(train['Parch'],train['Pclass'], margins=True, margins_name="Total").style.background_gradient(cmap='summer_r')
sns.barplot('Parch','Survived',data=train)
display(test[test['Fare'].isnull()])

display(test['Fare'].isnull().sum())
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger



med_fare = test.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]



test['Fare'] = test['Fare'].fillna(med_fare)
train.groupby('Survived').Fare.describe()
print('Highest Fare was:',train['Fare'].max())

print('Lowest Fare was:',train['Fare'].min())

print('Average Fare was:',train['Fare'].mean())
f,ax=plt.subplots(1,3,figsize=(20,8))

sns.distplot(train[train['Pclass']==1].Fare,ax=ax[0])

ax[0].set_title('Fares in Pclass 1')

sns.distplot(train[train['Pclass']==2].Fare,ax=ax[1])

ax[1].set_title('Fares in Pclass 2')

sns.distplot(train[train['Pclass']==3].Fare,ax=ax[2])

ax[2].set_title('Fares in Pclass 3')

plt.show()
sns.lmplot(x='Age', y='Fare', hue='Survived', data=train, fit_reg=False, scatter_kws={'alpha':0.5});
train['Cabin'] = train.drop(['Cabin'], inplace=True, axis=1)

test['Cabin'] = test.drop(['Cabin'], inplace=True, axis=1)
sns.distplot(train.Fare)
sns.distplot(train.Age.dropna())
plt.figure(figsize=(7, 6))

sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')
plt.figure(figsize=(7, 6))

sns.boxplot(x='Sex', y='Age', data=train)
plt.figure(figsize=(8, 7))

sns.boxplot(x='Sex', y='Age', data=train, hue='Survived')
sns.catplot(x='Pclass', y='Fare', data=train, kind='bar')
sns.catplot(x='Pclass', y='Survived', hue='Sex', data=train, kind='bar')
sns.catplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train, kind='bar')
sns.pairplot(numeric_cols_train)
columns = ["Pclass", "Sex", "SibSp", "Parch"]



X = pd.get_dummies(train[columns])

y = train["Survived"]



X_test = pd.get_dummies(test[columns])
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)
y_pred = model.predict(X_test)
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})

submission.to_csv('gender_submission.csv', index=False)
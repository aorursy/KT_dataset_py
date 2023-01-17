# import libraries



# for data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# for visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# for machine learning tasks

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%time df_train = pd.read_csv('../input/titanic/train.csv')

%time df_test = pd.read_csv('../input/titanic/test.csv')

# check the shape of train data

print(df_train.shape)
# look at 10 record of train data

df_train.head(10)
# train data info

df_train.info()
# checking null values

df_train.isna().sum()
# heatmap of null values

sns.heatmap(df_train.isna(), cbar=False)
# fill age with average value and drop cabin column

df_train.drop('Cabin', axis=1, inplace=True)

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

sns.heatmap(df_train.isnull(), cbar=False)
plt.figure(figsize=(18,6))

sns.heatmap(df_train.corr().abs(), cmap='plasma')

plt.title('Correlation Heatmap')
plt.figure(figsize=(18,6))

sns.clustermap(df_train.corr().abs(), cmap='plasma')

plt.title('Cluster Heatmap')
men = df_train[df_train['Sex']=='male']

plt.figure(figsize=(12,6))

sns.swarmplot(x='Pclass', y='Age', hue='Survived', data=men)

plt.title('Pclass vs. Age for Men')
women = df_train[df_train['Sex']!='male']

plt.figure(figsize=(12,6))

sns.swarmplot(x='Pclass', y='Age', hue='Survived', data=women)

plt.title('PClass vs. Age for Women')
plt.figure(figsize=(18,6))

sns.barplot(df_train['Age'], df_train['Sex'], hue=df_train['Survived'])
fc = sns.FacetGrid(df_train, row='Embarked', size=4, aspect=2)

fc.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='Set1')
# load test data

df_test = pd.read_csv('../input/titanic/test.csv')
df_test.shape
df_test.info()
df_test.isna().sum()
# fill age with average value and drop cabin column

df_test.drop('Cabin', axis=1, inplace=True)

df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())

sns.heatmap(df_test.isnull(), cbar=False)
le = LabelEncoder()

for each in df_train.columns:

    df_train[each] = le.fit_transform(df_train[each].astype(str))



for each in df_test.columns:

    df_test[each] = le.fit_transform(df_test[each].astype(str))
df_train.head()
X_train = df_train.drop(["Survived", "PassengerId", "Name"], axis=1)

y_train = df_train["Survived"]

X_test  = df_test.drop(["PassengerId", "Name"], axis=1)
clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)
Y_pred = clf.predict(X_test)

Y_pred
submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission_RandomForest.csv', index=False)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(20,5)})

import re

from sklearn.model_selection import train_test_split

from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
# for column in df_train:
#     print(df_train[column].unique())

# print(df_train['Cabin'].unique())

# df_train['HEHE'] = np.nan
# df_train = df_train[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']]
# df_test = df_test[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']]

# df_train.describe()

# # Turning cabin number into Deck
# cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
# df_train['Deck']=df_train['Cabin'].map(lambda x: needle_in_haystack(x, cabin_list))

# df_train.head(10)
# df_train['Deck'] = df_train['Deck'].fillna('NA')
# print(df_train['Deck'].unique())
fig, axs = plt.subplots(ncols=4, figsize=(20,5))
sns.countplot(x="Pclass", data=df_train, ax=axs[0])
sns.countplot(x="Sex", data=df_train, ax=axs[1])
sns.countplot(x="SibSp", data=df_train, ax=axs[2])
sns.countplot(x="Embarked", data=df_train, ax=axs[3])
fig, axs = plt.subplots(ncols=4, figsize=(20,5))
sns.countplot(x="Pclass", hue="Survived", data=df_train, ax=axs[0])
sns.countplot(x="Sex", hue="Survived", data=df_train, ax=axs[1])
sns.countplot(x="SibSp", hue="Survived", data=df_train, ax=axs[2])
sns.countplot(x="Embarked", hue="Survived", data=df_train, ax=axs[3])
df_train = df_train[['PassengerId', 'Name', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']]
df_test = df_test[['PassengerId', 'Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']]
df_train.head()
# fixing 'Age' in both, 'Fare' in train, 'Embarked' in test

dfs = [df_train, df_test]

# removing all NULL values in 'Fare'
for df in dfs:
    df['Fare'] = df['Fare'].fillna(df_train['Fare'].median())
    df['CategoricalFare'] = pd.qcut(df_train['Fare'], 4)
    
# removing all NULL values in 'Embarked'
for df in dfs:
    # df['Embarked'] = df['Embarked'].fillna('X')
    df['Embarked'] = df['Embarked'].fillna('S')

# removing all NULL values in 'Age'
for df in dfs:
    avg = df['Age'].mean()
    std = df['Age'].std()
    df['Age'] = df['Age'].fillna(np.random.randint(avg-std, avg+std))
    df['CategoricalAge'] = pd.cut(df['Age'], 5)

for df in dfs:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
for df in dfs:
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""    
    
for df in dfs:
    df['Title_x'] = df['Name'].apply(get_title)
    
    df['Title_x'] = df['Title_x'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title_x'] = df['Title_x'].replace('Mlle', 'Miss')
    df['Title_x'] = df['Title_x'].replace('Ms', 'Miss')
    df['Title_x'] = df['Title_x'].replace('Mme', 'Mrs')


# df_train.isnull().mean()
# df_test.isnull().mean()
df_train.head()
# creating mappings
for df in dfs:
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2, 'X':3})

    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)
    
    df.loc[ df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age'] = 4
    df['Age'] = df['Age'].astype(int)
    
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title_x'].map(title_mapping)
    # df['T'] = df['Title'].fillna(0).astype(int)

    


# feature extraction
df_train = df_train[['Survived', 'Title', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'IsAlone']]
df_test = df_test[['Pclass', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'IsAlone']]
df_train.head()
sns.factorplot(x="Sex" ,hue="IsAlone", col="Survived", kind="count", data=df_train)
sns.factorplot(x="Sex" ,hue="Pclass", col="Survived", kind="count", data=df_train)
# rules
# if Sex == 0 and (Pclass == 1 or Pclass == 2): Survived = 1
# elif Sex == 1 and Pclass == 3: Survived = 0
# else: //consider (Sex = 0 and Pclass == 3) and (Sex = 1 and Pclass = 1 or Pclass = 2 )
#     if title
# dft1 = df_train.loc[(df_train.Sex != 0) | (df_train.Pclass == 3), ['Survived', 'Title', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'IsAlone']]
# dft2 = dft1.loc[(dft1.Sex != 1) | (dft1.Pclass != 3), ['Survived', 'Title', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'IsAlone']]
# dft2.head()
# dft2_female = dft2.loc[(dft2.Sex == 0), ['Survived', 'Title', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'IsAlone']]
# sns.factorplot(x="Embarked", col="Survived", kind="count", data=dft2_female)
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df_train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(dft2[['Title', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'IsAlone']], dft2['Survived'], test_size=0.2)

# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)
# from sklearn import tree, svm
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier

# dt = tree.DecisionTreeClassifier()
# dt.fit(X_train, y_train)

# rf = RandomForestClassifier(n_estimators=10, max_depth=8, min_samples_split=2, random_state=0)
# rf.fit(X_train, y_train)

# svc = svm.SVC()
# svc.fit(X_train, y_train)

# mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# mlp.fit(X_train, y_train)

# xgb = XGBClassifier()
# xgb.fit(X_train, y_train)
# print("dt: ", dt.score(X_test, y_test))
# print("rf: ", rf.score(X_test, y_test))
# print("svc: ", svc.score(X_test, y_test))
# print("mlp: ", mlp.score(X_test, y_test))
# print("xgb: ", xgb.score(X_test, y_test))
# split data
X_train, X_test, y_train, y_test = train_test_split(df_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'IsAlone']], df_train['Survived'], test_size=0.25)
# X_train, X_test, y_train, y_test = train_test_split(dft2[['Sex', 'Age', 'IsAlone', 'Pclass', 'Fare', 'Title']], dft2['Survived'], test_size=0.2)

# make classifiers
dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=10, max_depth=8, min_samples_split=2, random_state=0)
rf.fit(X_train, y_train)
svc = svm.SVC()
svc.fit(X_train, y_train)
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(X_train, y_train)
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# get results
print(dt.score(X_test, y_test))
print(rf.score(X_test, y_test))
print(svc.score(X_test, y_test))
print(mlp.score(X_test, y_test))
print(xgb.score(X_test, y_test))
# df_test = df_test[['Sex', 'Age', 'IsAlone', 'Pclass', 'Fare', 'Title']]
df_test = df_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'IsAlone']]

predictions = xgb.predict(df_test)
df = pd.read_csv('../input/test.csv')
PassengerId = df['PassengerId']
submission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
submission.to_csv("submission10.csv", index=False)
submission




# Reference
# https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
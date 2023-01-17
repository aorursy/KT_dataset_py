# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # plotting graphs
import seaborn as sns # Seaborn for plotting and styling
import math
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
combine = [train, test]
train.head()
print(train.columns.values)
train.columns
#def count_analysis()
##def create_subplot()
##def draw_graphs()

cols = ['Pclass', 'Sex', 'SibSp','Survived',
       'Parch', 'Embarked']

fig,ax = plt.subplots(math.ceil(len(cols)/3),3,figsize=(20, 12))
ax = ax.flatten()
for a,s in zip(ax,cols):
    sns.countplot(x =s,data = train,palette = "bright",ax =a)
#def count_analysis()
##def create_subplot()
##def draw_graphs()

cols = ['Pclass', 'Sex', 'SibSp',
       'Parch', 'Embarked']

fig,ax = plt.subplots(math.ceil(len(cols)/3),3,figsize=(20, 12))
ax = ax.flatten()
for a,s in zip(ax,cols):
    sns.barplot(x =s,y="Survived",data = train,palette = "bright",ax =a)
cols = ['Pclass', 'Sex', 'SibSp',
       'Parch', 'Embarked']

fig,ax = plt.subplots(math.ceil(len(cols)/3),3,figsize=(20, 12))
ax = ax.flatten()
for a,s in zip(ax,cols):
    sns.countplot(x=s, hue="Survived", data=train,palette = "bright",ax =a)
x = pd.Series(train["Age"])
fig,ax = plt.subplots(figsize=(20, 10)) 
sns.distplot(x, color="y",ax = ax)
train.corr()
cols = ['Pclass', 'SibSp','Survived','Fare',
       'Parch']
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train[cols].astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

print("TRAIN DATA\n")
train.info()
print("\n===================================================\n")
print("TEST DATA\n")
test.info()
train.describe()
# check for na values
for col in train.columns:
    print("No. of Na values in " + str(col) + " " + str(len(train[train[col].isnull()])) +'\n')
imputed_age = float(train['Age'].dropna().median())
imputed_embarked = train['Embarked'].dropna().mode()[0]
print("imputed_age : ",imputed_age)
print("imputed_embarked : ",imputed_embarked)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(imputed_embarked)
    dataset['Age'] = dataset['Age'].fillna(imputed_age)
train.describe(include=['O'])
#########
# QC

# check for na values
for col in train.columns:
    print("No. of Na values in " + str(col) + " " + str(len(train[train[col].isnull()])) +'\n')
combine=[train,test]
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
    dataset['Age'] = dataset['Age'].astype(int)
    


train.head()
train = train.drop(['Ticket', 'Cabin','Name'], axis=1)
test = test.drop(['Ticket', 'Cabin','Name'], axis=1)
test['Fare'].fillna(test['Fare'].dropna().mean(), inplace=True)
# test.head()
train['FareBand'] = pd.qcut(train['Fare'], 5)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
combine = [train,test]
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <=  21.679), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <=  39.688), 'Fare']   = 3
    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)
combine = [train,test]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train.head()

train_df = train
test_df = test
X_train = train_df.drop("Survived", axis=1)
X_train = X_train.drop("PassengerId", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
X_train.head()
X_test.columns
X_train.info()
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission_v2.csv', index=False)


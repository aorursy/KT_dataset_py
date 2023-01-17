# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.impute import SimpleImputer

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Get data

test_data = pd.read_csv('../input/titanic/test.csv')

df = pd.read_csv('../input/titanic/train.csv')

df.describe()
df.isnull().sum()

df['Embarked'].describe()

data = [test_data, df]

for d in data:

    d['Embarked'] = d['Embarked'].fillna('S')

    d.isnull().sum()
data = [test_data, df]

for d in data:

    d['Age'] = d['Age'].fillna(d['Age'].mean())

    d.isnull().sum()
df = df.drop(['Cabin','Ticket','PassengerId'], axis = 1)

df
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,4))

women = df[df['Sex'] == 'female']

men = df[df['Sex'] == 'male']

ax = sns.distplot(a = women[women['Survived'] == 1].Age.dropna(), kde = False, 

                 label = 'Survived', ax = axes[0], bins=20)

ax = sns.distplot(a = women[women['Survived'] == 0].Age.dropna(), kde = False, 

                 label = 'Not Survived', ax = axes[0], bins=50)

ax.legend()

ax.set_title('Women')

ax = sns.distplot(a = men[men['Survived'] == 1].Age.dropna(), kde = False, 

                 label = 'Survived', ax = axes[1], bins=20)

ax = sns.distplot(a = men[men['Survived'] == 0].Age.dropna(), kde = False, 

                 label = 'Not Survived', ax = axes[1], bins=50)

ax.legend()

ax.set_title('Men')

plt.show()
fg = sns.FacetGrid(df, row='Embarked')

fg.map(sns.pointplot, 'Sex', 'Survived',

               palette=None,  order=None, hue_order=None )
sns.barplot(x=df['Pclass'], y=df['Survived'])

plt.show()
fg2 = sns.FacetGrid(df, col='Survived', row='Pclass')

fg2.map(plt.hist, 'Age', alpha=.5, bins=20)

data = [test_data, df]

for d in data:

    d['relatives'] = d['SibSp'] + d['Parch']

    d['relatives'].value_counts()
data = [test_data, df]

for d in data:

    d = d.drop(['SibSp','Parch'], axis = 1)

    d.info()
data = [test_data, df]

for d in data:

    d['Age'] = d['Age'].astype(int)

    d['Fare'] = d['Fare'].fillna(0)

    d['Fare'] = d['Fare'].astype(int)

    d.info()
data = [test_data, df]

for d in data:

    d['Embarked'] = d['Embarked'].astype('category')

    d['embarked_cat'] = d['Embarked'].cat.codes

    d = d.drop(['Embarked'],axis=1)

    d.head()
data = [test_data, df]

for d in data:

    d['Sex'] = d['Sex'].astype('category')

    d['sex_cat'] = d['Sex'].cat.codes

    d = d.drop(['Sex','Name'],axis=1)

    print(d)





test_data = test_data.drop(['Name','Cabin','Parch','Ticket','SibSp','Sex','Embarked'],axis=1)

df = df.drop(['Sex','Name','Embarked','SibSp','Parch'],axis=1)

df
X_train = df.drop(['Survived'],axis=1)

y_train = df['Survived']

X_test = test_data.drop(['PassengerId'],axis=1)

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

knn_accuracy = knn.score(X_train, y_train)

print(knn_accuracy)
linreg = LinearRegression()

linreg.fit(X_train,y_train)

y_pred = linreg.predict(X_test)

linreg_accuracy = linreg.score(X_train,y_train)

print(linreg_accuracy)
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred = linreg.predict(X_test)

logreg_accuracy = logreg.score(X_train,y_train)

print(logreg_accuracy)
tree = DecisionTreeClassifier()

tree.fit(X_train,y_train)

y_pred = tree.predict(X_test)

tree_accuracy = tree.score(X_train,y_train)

print(tree_accuracy)
rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

rf_accuracy = rf.score(X_train,y_train)

print(rf_accuracy)
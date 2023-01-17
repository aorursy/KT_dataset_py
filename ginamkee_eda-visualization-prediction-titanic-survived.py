# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



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
gd = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

dt = pd.read_csv('/kaggle/input/titanic/train.csv')

dtt = pd.read_csv('/kaggle/input/titanic/test.csv')
f,ax=plt.subplots(1,2,figsize=(18,8))

dt['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=dt,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
dt[dt['Age'] < 50].plot.hexbin(x='Age', y='Survived', gridsize=15)
data = pd.concat([dt['Age'], dt['Survived']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='Survived', y="Age", data=data)
ax = sns.countplot(x="Sex", hue='Survived', data=dt)
dt['Pclass'].value_counts().sort_index().plot.bar()
ax = sns.countplot(x="Pclass", hue='Sex', data=dt)
total = dt.isnull().sum().sort_values(ascending=False)

percent = (dt.isnull().sum()/dt.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



percent_data = percent.head(20)

percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)

plt.ylabel("Count", fontsize = 20)

plt.title("Total Missing Value (%)", fontsize = 20)
numeric_features = dt.select_dtypes(include=[np.number])

numeric_features.columns
dt[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
grid = sns.FacetGrid(dt, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
combine = [dt, dtt]
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

dt.head()
plt.figure(figsize=(15,8))

sns.distplot(dt['Age']);
import missingno as msno

msno.matrix(dt.sample(150))
sns.pointplot('Pclass','Survived',hue='Sex',data=dt)
ax = sns.violinplot(x="Pclass", y="Age", hue="Survived",

                    data=dt, palette="muted", split=True)
dt['Age'] = dt['Age'].fillna(dt['Age'].mean())
dt['AgeBand'] = pd.cut(dt['Age'], 5)

dt[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
dt = dt.drop(['Ticket'], axis=1)
dt['Embarked'] = dt['Embarked'].fillna(fre)
dt['Embarked'] = dt['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
total = dt.isnull().sum().sort_values(ascending=False)

percent = (dt.isnull().sum()/dt.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



percent_data = percent.head(20)

percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)

plt.ylabel("Count", fontsize = 20)

plt.title("Total Missing Value (%)", fontsize = 20)
dt = dt.drop(['AgeBand'], axis=1)
dt['Title'] = dt.Name.str.extract('([A-Za-z]+)\.', expand=False)



pd.crosstab(dt['Title'], dt['Sex'])
dt['Title'] = dt['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



dt['Title'] = dt['Title'].replace('Mlle', 'Miss')

dt['Title'] = dt['Title'].replace('Ms', 'Miss')

dt['Title'] = dt['Title'].replace('Mme', 'Mrs')

dt[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

dt['Title'] = dt['Title'].map(title_mapping)

dt['Title'] = dt['Title'].fillna(0)



dt.head()
dt = dt.drop(['Cabin', 'Name'], axis=1)
dtt = dtt.drop(['Cabin', 'Ticket'], axis=1)
dtt['Age'] = dtt['Age'].fillna(dtt['Age'].mean())
dtt['Title'] = dtt.Name.str.extract('([A-Za-z]+)\.', expand=False)



pd.crosstab(dtt['Title'], dtt['Sex'])
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

dtt['Title'] = dtt['Title'].map(title_mapping)

dtt['Title'] = dtt['Title'].fillna(0)



dtt.head()
dtt = dtt.drop(['Name'], axis=1)
dtt['Embarked'] = dtt['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
dtt['Fare'] = dtt.fillna(dtt['Fare'].mean())
X_train = dt.drop(['Survived', 'PassengerId'], axis=1)

Y_train = dt['Survived']
X_test = dtt.drop(['PassengerId'], axis=1).copy()
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
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
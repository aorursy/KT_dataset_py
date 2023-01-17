# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt
train_df = pd.read_csv('train.csv')

test_df = pd.read_csv('test.csv')

combine = [train_df, test_df]
train_df.head(10)
train_df.info()
train_df.Cabin.value_counts()
train_df.describe() #all numeric columns
train_df.describe(include=['O']) # all object columns
train_df.Pclass.value_counts() #counts of unique values.
train_df[['Pclass', 'Survived']].groupby('Pclass').mean()
train_df[['Sex', 'Survived']].groupby('Sex').mean()
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0})

train_df.head()
train_df[['Name', 'Survived']]
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract('(\w*)\.', expand=False)

train_df.head()
train_df.Title.value_counts()
train_df[['Title', 'Survived']].groupby('Title').mean()
pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



    vc = dataset.Title.value_counts()

    topTitles = vc.index[:4].values

    rareTitles = vc.index[4:].values

    

    dataset['Title'] = dataset['Title'].replace(rareTitles, 'Rare')

    

    i = 0

    for title in np.append(topTitles, 'Rare'):

        dataset.loc[(dataset['Title'] == title), 'Title'] = i

        i += 1

        

    dataset['Title'] = dataset['Title'].astype(int)

    

train_df.Title.value_counts()
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
fig = plt.figure();

ax = fig.add_subplot(111)

ax.hist(train_df.Age.dropna(), bins=80)
#fig = plt.figure()

#ax = fig.add_subplot(121)

#ax.hist(train_df.loc[train_df['Survived']==1].Age.dropna(), bins=20)

#ax = fig.add_subplot(122)

#ax.hist(train_df.loc[train_df['Survived']==0].Age.dropna(), bins=20)

plt.show()
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)

plt.show()
# g = sns.FaceGrid(train_df, col)

# plt.scatter(train_df.Age, train_df.Fare)

fig = plt.figure();

ax = fig.add_subplot(111)

ax.scatter(train_df.Fare, train_df.Age)

plt.show()
g = sns.FacetGrid(train_df, col='Sex')

g.map(plt.hist, 'Age', bins=20)

plt.show()
g = sns.FacetGrid(train_df, col='Pclass')

g.map(plt.hist, 'Age', bins=20)

plt.show()
g = sns.FacetGrid(train_df, col='Title')

g.map(plt.hist, 'Age', bins=20)

plt.show()
# 利用Title补全age字段

for dataset in combine:

    titles = dataset.Title.unique()

    guess_ages = {}

    for title in titles:

        guess_ages[title] = dataset.loc[train_df['Title'] == title].Age.median()

    for title in titles:

        dataset.loc[dataset.Age.isnull() & (dataset.Title == title), 'Age'] = guess_ages[title] 
fig = plt.figure()

ax = fig.add_subplot(1,2,1)

ax.hist(train_df.SibSp)

ax = fig.add_subplot(1,2,2)

ax.hist(train_df.Parch)

plt.show()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
for dataset in combine:

    dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)



train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]
# 清理无用字段

train_df = train_df.drop(['Cabin', 'Ticket', 'Name', 'SibSp', 'Parch', 'PassengerId', 'FamilySize'], axis=1)

test_df = test_df.drop(['Cabin', 'Ticket', 'Name', 'SibSp', 'Parch', 'FamilySize'], axis=1)

combine = [train_df, test_df]
test_df.head()
test_df.info()
train_df.head()
train_df.info()
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
x_train = train_df.drop('Survived', axis=1)

y_train = train_df['Survived']

x_test = test_df.drop("PassengerId", axis=1).copy()

x_train.shape, y_train.shape, x_test.shape
logReg = LogisticRegression()

logReg.fit(x_train, y_train)

log_pred = logReg.predict(x_test)

acc_log = round(logReg.score(x_train, y_train) * 100, 2)

acc_log
svc = SVC()

svc.fit(x_train, y_train)

svc_pred = svc.predict(x_test)

acc_svc = round(svc.score(x_train, y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train, y_train)

knn_pred = knn.predict(x_test)

acc_knn = round(knn.score(x_train, y_train) * 100, 2)

acc_knn
gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

gaussian_pred = gaussian.predict(x_test)

acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)

acc_gaussian
linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

linear_svc_pred = linear_svc.predict(x_test)

acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)

acc_linear_svc
perceptron = Perceptron()

perceptron.fit(x_train, y_train)

prec_pred = perceptron.predict(x_test)

acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)

acc_perceptron
sgd = SGDClassifier()

sgd.fit(x_train, y_train)

sgd_pred = sgd.predict(x_test)

acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)

acc_sgd
decision_tree = DecisionTreeClassifier()

decision_tree.fit(x_train, y_train)

dt_pred = decision_tree.predict(x_test)

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

rf_pred = random_forest.predict(x_test)

# random_forest.score(x_train, y_train)

acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)

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
submission = pd.DataFrame({

    "PassengerId": test_df["PassengerId"],

    "Survived": dt_pred

})
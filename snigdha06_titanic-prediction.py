# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Imports

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# get train & test csv files as DataFrame

train_df=pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
# preview the data

train_df.head(10)
train_df.shape, test_df.shape
train_df.info()
train_df.describe()
train_df.describe(include=['O'])
train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by = 'Survived',ascending=False)
train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by = 'Survived',ascending=False)
train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by = 'Survived',ascending=False)
train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by = 'Survived',ascending=False)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
train_df = train_df.drop(['Ticket', 'Cabin','Name','PassengerId'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin','Name'], axis=1)
"After", train_df.shape, test_df.shape
train_df = train_df.drop(['Fare'], axis=1)

test_df = test_df.drop(['Fare'], axis=1)
train_df.Age.replace(np.NaN,train_df.Age.mean(),inplace=True)

test_df.Age.replace(np.NaN,test_df.Age.mean(),inplace=True)
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
test_df['AgeBand'] = pd.cut(test_df['Age'], 5)
for dataset in [train_df]:   

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()
for dataset in [test_df]:   

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

test_df.head()
train_df=train_df.drop(['AgeBand'],axis=1)

test_df=test_df.drop(['AgeBand'],axis=1)
train_df.head(),test_df.head()
for dataset in [train_df,test_df]:   

    dataset.loc[ dataset['Sex'] == 'male', 'Sex'] = 0

    dataset.loc[ dataset['Sex'] == 'female', 'Sex'] = 1

train_df.head(), test_df.head()
for dataset in [train_df,test_df]:   

    dataset.loc[ dataset['Embarked'] == 'S', 'Embarked'] = 0

    dataset.loc[ dataset['Embarked'] == 'Q', 'Embarked'] = 1

    dataset.loc[ dataset['Embarked'] == 'C', 'Embarked'] = 2

train_df.head()
freq_port_train = train_df.Embarked.dropna().mode()[0]

freq_port_test = test_df.Embarked.dropna().mode()[0]
train_df['Embarked'].fillna(freq_port_train)

train_df.Embarked.replace(np.NaN,freq_port_train,inplace=True)

test_df['Embarked'].fillna(freq_port_test)

test_df.Embarked.replace(np.NaN,freq_port_test,inplace=True)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]
X_train.shape,Y_train.shape,test_df.shape
X_test=test_df.drop("PassengerId", axis=1).copy()
X_train.isnull().values.ravel().sum()

df_null =X_train.isnull().unstack()

df_null[df_null]

t = df_null[df_null]



X_train.Embarked.isnull().sum()

X_train.Embarked.replace(np.NaN,freq_port_train,inplace=True)

X_train.Embarked.isnull().sum(), X_test.Embarked.isnull().sum()
logreg=LogisticRegression()

logreg.fit(X_train,Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines



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
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest



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
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })
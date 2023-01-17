import pandas as pd

import numpy as np



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

# combine = [train_df, test_df]

print(train_df.columns)

train_df.head()
train_df.tail()
print('Train Info:')

train_df.info()

print('\n\nTest Info:')

test_df.info()
train_df.describe()
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
grid = sns.FacetGrid(train_df, col='Survived')

grid.map(plt.hist, 'Age', bins=20)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
train_df = train_df.drop(["PassengerId", "Ticket", "Cabin", "Name", "Embarked", "Fare", "SibSp", "Parch"], axis=1)

train_df['Age'] = train_df['Age'].fillna(0)

train_df['Age'] = train_df['Age'].astype(int)



test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
# Convert String female = 1 and male = 0

train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

X_train = train_df.drop(["Survived"], axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()



X_train.head()


X_test = X_test.drop(["Name", "Embarked", "Fare", "SibSp", "Parch"], axis=1)

X_test['Sex'] = X_test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

X_test['Age'] = X_test['Age'].fillna(0)

X_test['Age'] = X_test['Age'].astype(int)

X_test.head()

# Logistic Regression



# machine learning

from sklearn.linear_model import LogisticRegression



model_lr = LogisticRegression()

model_lr.fit(X_train, Y_train)

Y_pred = model_lr.predict(X_test)

acc_log = model_lr.score(X_train, Y_train)

acc_log
# Support Vector Machines

from sklearn.svm import SVC





svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = svc.score(X_train, Y_train)

acc_svc
# K-Nearest Neighbors



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = knn.score(X_train, Y_train)

acc_knn
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB





gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = gaussian.score(X_train, Y_train)

acc_gaussian
# Perceptron

from sklearn.linear_model import Perceptron





perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = perceptron.score(X_train, Y_train)

acc_perceptron
# Linear SVC

from sklearn.svm import LinearSVC





linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = linear_svc.score(X_train, Y_train)

acc_linear_svc
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier





sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = sgd.score(X_train, Y_train)

acc_sgd
# Decision Tree

from sklearn.tree import DecisionTreeClassifier





decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = decision_tree.score(X_train, Y_train)

acc_decision_tree
# Random Forest

from sklearn.ensemble import RandomForestClassifier





random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = random_forest.score(X_train, Y_train)

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
submission.to_csv('submission.csv', index=False)

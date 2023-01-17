#import libraries
import pandas as pd
import numpy as np
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

#read data 
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.info()
train.Cabin.unique()
train.describe()
train.describe(include = ['O'])
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False).plot.bar(x = 'Pclass', y = 'Survived')
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False).plot.bar(x = 'Sex', y = 'Survived')
train[["Age", "Survived"]].groupby(['Age'], as_index = False).mean().sort_values(by = 'Age').plot.bar(x = 'Age', y = 'Survived', figsize = (25, 6))
grid = sns.FacetGrid(train, col = 'Survived', size = 5)
grid.map(plt.hist, 'Age', bins = 40)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2, aspect=2)
grid.map(plt.hist, 'Sex', alpha=0.7, bins=4)
grid.add_legend();
grid = sns.FacetGrid(train, col='Embarked', size=3, aspect=2)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
grid = sns.FacetGrid(train, col='Embarked', row='Survived', size=2, aspect=2)
grid.map(sns.barplot, 'Sex','Fare', alpha=0.7, ci=None)
grid.add_legend();
print ("Before", train.shape, test.shape)
train = train.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis = 1)
test = test.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis = 1)
print("After",train.shape, test.shape )
train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train.head()
train.Age.value_counts().sort_values(ascending = False).head(5)
test.Age.value_counts().sort_values(ascending = False).head(5)
train.Age = train.Age.fillna(24)
test.Age = test.Age.fillna(24)
test.Age.unique()
train.head()
train['Embarked'] = train['Embarked'].map( {'S': 1, 'C': 2, 'Q':3, np.nan: 0, 24: 0} ).astype(int)
test['Embarked'] = test['Embarked'].map( {'S': 1, 'C': 2, 'Q':3, np.nan: 0, 24: 0} ).astype(int)
train.head()
train['FamilySize'] = pd.DataFrame(train.SibSp + train.Parch)
train.describe()
test['FamilySize'] = pd.DataFrame(test.SibSp + test.Parch)
test.describe()
train= train.drop(['SibSp', 'Parch'], axis = 1)
test= test.drop(['SibSp', 'Parch'], axis = 1)
train.head()
train.Age.unique()
test.Fare.value_counts().head()
test.Fare = test.Fare.fillna(7.75)
x_train = train.drop("Survived", axis = 1)
y_train = train.Survived
x_test = test
x_train.shape,y_train.shape, x_test.shape
x_test.info()
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
acc_log
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
acc_knn
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
acc_svc
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
acc_gaussian
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_test)
acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)
acc_perceptron
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_test)
acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)
acc_linear_svc
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
acc_random_forest
y_pred.astype(np.int32)
np.savetxt("output.csv", y_pred)
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_test)
acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)
acc_sgd
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
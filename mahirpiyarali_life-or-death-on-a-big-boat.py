#processing

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

#visualizations

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



test_df = pd.read_csv('../input/test.csv')

train_df = pd.read_csv('../input/train.csv')

combine = [train_df, test_df]

#combine[:]

fares = []

for fr in train_df["Fare"]:

    fares.append(fr)

#print(fares)
train_df.info()
train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)



train_df.head() #this line of code simply prints the first 5 values in the dataset
train_df = train_df.drop(['Ticket'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1)

train_df.head()
train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

train_df.head()
#looking at correlation between class and survival

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df = train_df.drop(['Parch'], axis=1)

test_df = test_df.drop(['Parch'], axis=1)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

g = sns.FacetGrid(train_df,col='Survived')

g.map(plt.hist, 'Age', bins=20)
freq_age = train_df.Age.dropna().mode()[0] #this code finds us the most common age point

train_df['Age'] = train_df['Age'].fillna(freq_age) 

test_df['Age'] = test_df['Age'].fillna(freq_age)



train_df[:10]
freq_fare = train_df.Fare.dropna().mode()[0] #this code finds us the most common embarkation point

#freq_port

train_df['Fare'] = train_df['Fare'].fillna(freq_fare) 

test_df['Fare'] = test_df['Fare'].fillna(freq_fare)
embr = sns.FacetGrid(train_df, col='Survived')

embr.map(plt.hist, 'Embarked', bins=20)

#train_df.head()
train_df = train_df.drop(['Embarked'], axis=1)

test_df = test_df.drop(['Embarked'], axis=1)
train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

test_df['Sex'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
X_train = train_df.drop("Survived", axis=1)

X_train = X_train.drop("PassengerId", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1)

#X_train[:10]

#Y_train[:10]

X_test[:10]
#k nearest neighbors

#knn = KNeighborsClassifier(n_neighbors = 3)

#knn.fit(X_train, Y_train)

#Y_pred = knn.predict(X_test)

#acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

#acc_knn
#random forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
#d-tree

#decision_tree = DecisionTreeClassifier()

#decision_tree.fit(X_train, Y_train)

#Y_pred = decision_tree.predict(X_test)

#acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

#acc_decision_tree
#stochastic dradient descent

#sgd = SGDClassifier()

#sgd.fit(X_train, Y_train)

#Y_pred = sgd.predict(X_test)

#acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

#acc_sgd
#linear svc

#linear_svc = LinearSVC()

#linear_svc.fit(X_train, Y_train)

#Y_pred = linear_svc.predict(X_test)

#acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

#acc_linear_svc
#perceptron

#perceptron = Perceptron()

#perceptron.fit(X_train, Y_train)

#Y_pred = perceptron.predict(X_test)

#acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

#acc_perceptron
#gaussian naive bayes

#gaussian = GaussianNB()

#gaussian.fit(X_train, Y_train)

#Y_pred = gaussian.predict(X_test)

#acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

#acc_gaussian
#support vector machines

#svc = SVC()

#svc.fit(X_train, Y_train)

#Y_pred = svc.predict(X_test)

#acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

#acc_svc
#logistic regression

#logreg = LogisticRegression()

#logreg.fit(X_train, Y_train)

#Y_pred = logreg.predict(X_test)

#acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

#acc_log
#models = pd.DataFrame({

#    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

#              'Random Forest', 'Naive Bayes', 'Perceptron', 

#              'Stochastic Gradient Decent', 'Linear SVC', 

#              'Decision Tree'],

#    'Score': [acc_svc, acc_knn, acc_log, 

#              acc_random_forest, acc_gaussian, acc_perceptron, 

#              acc_sgd, acc_linear_svc, acc_decision_tree]})

#models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)
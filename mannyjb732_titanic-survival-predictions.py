#import packages



import pandas as pd 

import numpy as np 

import seaborn as sn

import matplotlib.pyplot as plt 

%matplotlib inline 

#import test and train data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.info()
train.isnull().sum()
plt.figure(figsize=(12,8))

sn.heatmap(train.isnull(), yticklabels=False, cbar=False)

plt.show()
train.describe()


corr = train.corr()

plt.figure(figsize=(13.5,10))

sn.heatmap(corr, annot=True, cmap='seismic_r', linewidths=.5)

plt.show()

plt.figure(figsize=(12,8))

sn.countplot('Survived', data=train, palette='RdBu_r')

plt.title('Count of Passenger by Survived')

plt.show()
grid = sn.FacetGrid(train, row='Pclass', col='Sex', hue='Sex', size=3.5, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()

plt.show()
plt.figure(figsize=(13.5,7))

sn.boxplot('Embarked', 'Fare', data = train)

plt.title('Embarked by Fare')

plt.show()
plt.figure(figsize=(13.5, 7))

sn.boxplot('Embarked', 'Age', data=train)

plt.title('Emabrked by Age')

plt.show()
grid = sn.FacetGrid(train, row='Sex', col='Survived', hue='Sex', size=3.5, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins = 20)

grid.add_legend()

plt.show()
combined = [train, test]

for dataset in combined :

    dataset['Embarked'] = dataset['Embarked'].fillna('C')
for dataset in combined: 

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for dataset in combined:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
for dataset in combined: 

    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}

for dataset in combined: 

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



embark_map = {'S': 1, 'Q': 2, 'C': 3}

for dataset in combined:

    dataset['Embarked'] = dataset['Embarked'].map(embark_map)
train[['Title', 'Sex']].groupby('Title').count()


sn.factorplot('Title', data=train, kind='count', size=5, aspect=2)

plt.title('Title Distribution')

plt.show()
train.head()
test.head()
guess_ages = np.zeros((2,3))



guess_ages



for dataset in combined: 

    for i in range(0,2):

        for j in range(0,3):

            guess_df = dataset[(dataset['Sex']==i) & (dataset['Pclass']==j+1)]['Age'].dropna()

            

            age_guess = guess_df.median() 

            

            guess_ages[i,j] = int(age_guess/0.5 + 0.5)*0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)

        
train = train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 

test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1) 



test['Fare'] = test['Fare'].fillna(0)
train.isnull().sum()
grid = sn.FacetGrid(train, hue='Sex', size=6.5, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()

plt.show()
train.head()
test.head()
test.isnull().sum()
test.head()
#Manchine Learning Models 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
x_train = train.drop('Survived', axis=1)

y_train = train['Survived']



x_test = test.drop('PassengerId', axis=1).copy()
logreg = LogisticRegression()

logreg.fit(x_train, y_train)

Y_pred = logreg.predict(x_test)

acc_log = round(logreg.score(x_train, y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df['Correlation'] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
svc = SVC()

svc.fit(x_train, y_train)

Y_pred = svc.predict(x_test)

acc_svc = round(svc.score(x_train, y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train, y_train)

Y_pred = knn.predict(x_test)

acc_knn = round(knn.score(x_train, y_train) * 100, 2)

acc_knn
gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

Y_pred = gaussian.predict(x_test)

acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)

acc_gaussian
perceptron = Perceptron()

perceptron.fit(x_train, y_train)

Y_pred = perceptron.predict(x_test)

acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)

acc_perceptron
linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

Y_pred = linear_svc.predict(x_test)

acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)

acc_linear_svc
sgd = SGDClassifier()

sgd.fit(x_train, y_train)

Y_pred = sgd.predict(x_test)

acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)

acc_sgd
#Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(x_train, y_train)

Y_pred = decision_tree.predict(x_test)

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

Y_pred = random_forest.predict(x_test)

random_forest.score(x_train, y_train)

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
submission = pd.DataFrame({"Passenger": test["PassengerId"], "Survived": Y_pred})

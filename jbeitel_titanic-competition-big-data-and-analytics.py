#data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
# preview the data
train_df.head()
print(train_df.columns.values)
train_df.info()
print('_'*40)
test_df.info()
#turn categorical data for 'Sex' into binary dataset using 1s and 0s
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
#combine sibsp and parch
train_df["Family"] = train_df["SibSp"] + train_df["Parch"]
test_df["Family"] = test_df["SibSp"] + test_df["Parch"]
#drop columns that we aren't confident in
train_df.drop('Parch', axis = 1, inplace = True)
train_df.drop('SibSp', axis = 1, inplace = True)
train_df.drop('Ticket', axis = 1, inplace = True)
train_df.drop('Embarked', axis = 1, inplace = True)
train_df.drop('Cabin', axis = 1, inplace = True)
train_df.drop('Name', axis = 1, inplace = True)
train_df.drop('Age', axis = 1, inplace = True)

#drop columns that we aren't confident in
test_df.drop('Parch', axis = 1, inplace = True)
test_df.drop('SibSp', axis = 1, inplace = True)
test_df.drop('Ticket', axis = 1, inplace = True)
test_df.drop('Embarked', axis = 1, inplace = True)
test_df.drop('Cabin', axis = 1, inplace = True)
test_df.drop('Name', axis = 1, inplace = True)
test_df.drop('Age', axis = 1, inplace = True)
#better analysis than the family key feature
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['Family'] == 0, 'IsAlone'] = 1
#print finalized tail of dataset
train_df.head()
#print finalized tail of dataset
test_df.head()
#find correlations to find the survivability of different features
traincorr = train_df.corr(method='spearman')

traincorr.drop('PassengerId', axis = 1, inplace = True)
traincorr.drop('Pclass', axis = 1, inplace = True)
traincorr.drop('Sex', axis = 1, inplace = True)
#traincorr.drop('Fare', axis = 1, inplace = True)
traincorr.drop('Family', axis = 1, inplace = True)
traincorr.drop('IsAlone', axis = 1, inplace = True)

traincorr
fare_mean = np.mean(test_df['Fare'])
test_df['Fare'].fillna(fare_mean, inplace=True)
fare_mean = np.mean(train_df['Fare'])
train_df['Fare'].fillna(fare_mean, inplace=True)
test_df.info()
train_df.info()
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='IsAlone', ascending=False)
train_df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().sort_values(by='Family', ascending=False)
for val in train_df.columns:
    if val != 'Sex' and val != 'PassengerId' and val != 'Survived' and val != 'Pclass' and val != 'Fare': 
        train_df = train_df.drop([val], axis=1)
        test_df = test_df.drop([val], axis=1)
        
for val in test_df.columns:
    if val != 'Sex' and val != 'PassengerId' and val != 'Pclass' and val != 'Fare': 
        train_df = train_df.drop([val], axis=1)
        test_df = test_df.drop([val], axis=1)
X_train = train_df.drop(["Survived", 'PassengerId'], axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1)

X_train.shape, Y_train.shape, X_test.shape
#cross validation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split #split data into train and test set 
#X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size = .7, test_size = .3)

valid_X_train, valid_X_test, valid_Y_train, valid_Y_test = train_test_split(X_train, Y_train, train_size = .7, test_size = .3)

valid_X_train.shape, valid_X_test.shape, valid_Y_train.shape, valid_Y_test.shape
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(valid_X_train, valid_Y_train)
Y_pred = logreg.predict(valid_X_test)
acc_log = round(logreg.score(valid_X_train, valid_Y_train) * 100, 2)
acc_log
# Support Vector Machines
svc = SVC()
svc.fit(valid_X_train, valid_Y_train)
Y_pred = svc.predict(valid_X_test)
acc_svc = round(svc.score(valid_X_train, valid_Y_train) * 100, 2)
acc_svc
#KNearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(valid_X_train, valid_Y_train)
Y_pred = knn.predict(valid_X_test)
acc_knn = round(knn.score(valid_X_train, valid_Y_train) * 100, 2)
acc_knn
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(valid_X_train, valid_Y_train)
Y_pred = gaussian.predict(valid_X_test)
acc_gaussian = round(gaussian.score(valid_X_train, valid_Y_train) * 100, 2)
acc_gaussian
# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(valid_X_train, valid_Y_train)
Y_pred = linear_svc.predict(valid_X_test)
acc_linear_svc = round(linear_svc.score(valid_X_train, valid_Y_train) * 100, 2)
acc_linear_svc
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(valid_X_train, valid_Y_train)
Y_pred = decision_tree.predict(valid_X_test)
acc_decision_tree = round(decision_tree.score(valid_X_train, valid_Y_train) * 100, 2)
acc_decision_tree
#Decision Tree overfit test - SUCCESS
print((decision_tree.score(X_train, Y_train)*100), decision_tree.score(valid_X_train, valid_Y_train)*100)
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(valid_X_train, valid_Y_train)
Y_pred = random_forest.predict(valid_X_test)
random_forest.score(valid_X_train, valid_Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
#Random Forest overfit test - SUCCESS
print((random_forest.score(X_train, Y_train)*100), random_forest.score(valid_X_train, valid_Y_train)*100)
# AdaBoost Classifier
ada_boost = AdaBoostClassifier(n_estimators=300)
ada_boost.fit(valid_X_train, valid_Y_train)
Y_pred = ada_boost.predict(valid_X_test)
ada_boost.score(valid_X_train, valid_Y_train)
acc_ada_boost = round(ada_boost.score(valid_X_train, valid_Y_train) * 100, 2)
acc_ada_boost
final_pred = decision_tree.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': final_pred})

submission
submission.to_csv('submission.csv', index=False)

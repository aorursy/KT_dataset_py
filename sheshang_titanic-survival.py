# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.info()
test.info()

train_label = train[['PassengerId','Survived']]
train = train.drop('Survived', axis=1)
test_id = test[['PassengerId']]
combined = pd.concat([train, test], keys=['train', 'test'])
combined.info()
test_id.info()
# Any results you write to the current directory are saved as output.
#here we took all known embarked values and visualize based on fares and pclass.
#then we took unknown embarked values and print hor. line on based on their fare and pclass.
#we can see that it passes from the median of pclass 1 and embarked station 'C'.
#so wee will fill null embarked values with 'C
unknownEmbarked = tuple(combined.loc[combined['Embarked'].isnull(),'Fare'])
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(x='Embarked', y='Fare', hue='Pclass', data=combined[combined['Embarked'].notnull()]);
ax.hlines(y=unknownEmbarked, xmin=-1, xmax=3)
combined[['Embarked']] = combined[['Embarked']].fillna(value='C')
#now we are filling null fare values.
#the passenger having null fare value belong to pclass 3 and emabkerment station S.
#so we take fare median for all passengers having pclass 3 and 'S' embarkment station and assign to null fare.
unknownFare = combined.loc[(combined['Embarked'] == 'S') & (combined['Pclass'] == 3) & (combined['Fare'].notnull()), 'Fare'].median()
combined[['Fare']] = combined[['Fare']].fillna(value=unknownFare)
combined['Title'] = combined.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(combined['Title'], combined['Sex'])
combined['Title'] = combined['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Rare')
combined['Title'] = combined['Title'].replace(['Ms', 'Mlle'], 'Miss')
combined['Title'] = combined['Title'].replace('Mme', 'Mrs')

pd.crosstab(combined['Title'], combined['Sex'])
title_mapping = {"Mr" : 1, "Miss" : 2, "Master" : 3, "Mrs" : 4, "Rare" : 5}
combined["Title"] = combined["Title"].map(title_mapping)

sex_mapping = {"female" : 1, "male" : 2}
combined["Sex"] = combined["Sex"].map(sex_mapping)

embark_mapping = {"S" : 1, "C" : 2, "Q" : 3}
combined["Embarked"] = combined["Embarked"].map(embark_mapping)
combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
combined.info()
combined.head()
from fancyimpute import MICE, KNN, NuclearNormMinimization, SoftImpute
#dropping 'PassengerId', 'Name', 'Fare', 'Ticket', 'Cabin' bcz they are not helpfull.
#Dropping 'Fare' bcz it is repeatative measure of 'Embarked', 'Pclass', 'Age'(probably)

combined = combined.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis = 1)
combined_modified = MICE().complete(combined)
combined_modified = pd.DataFrame(data=combined_modified, columns=combined.columns)
combined_modified.info()
fig, axes = plt.subplots(1, 2)
(a, bins, patches) = axes[0].hist(combined.dropna()['Age'], bins=8, label='hst')
(b, bins, patches) = axes[1].hist(combined_modified['Age'], bins=8, label='hst')

print (a)
print (b)
#modeling and preparing datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

X_train = combined_modified[0:891]
#in following line ravel() function is to convert dataframe's column vector to 1d array, as 1d array is axpected in modeling algos
Y_train = np.ravel(train_label.drop("PassengerId", axis=1))
X_test = combined_modified[891:]
X_train.shape, Y_train.shape, X_test.shape
#logistic regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
coeff_df = pd.DataFrame(X_train.columns)
coeff_df.columns = ['Features']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])

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
    "PassengerId" : test_id["PassengerId"],
    "Survived" : Y_pred
})

submission.to_csv('submission.csv', index=False)
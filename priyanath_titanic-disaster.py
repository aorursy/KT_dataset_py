# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

sample_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

combine = [train, test]
train.head(5)

test.head(5)
train.info()

print('-'*30)

test.info()
train.describe()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
train[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
train[['Parch', 'Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
# Histogram betweeen Age and Survived

g=sns.FacetGrid(train, col = 'Survived')

g.map(plt.hist,'Age', bins = 20)
# Histogram between Survived and PClass

g1= sns.FacetGrid(train, col = 'Survived', row = 'Pclass' )

g1.map(plt.hist, 'Age', bins = 20)

g2 = sns.FacetGrid(train, col = 'Embarked')

g2.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', pallete = 'deep')

g2.add_legend()
g3 = sns.FacetGrid(train, col = 'Embarked', row = 'Survived')

g3.map(sns.barplot,'Sex', 'Fare')

g3.add_legend()
for dataset in combine:    

    #complete missing age with median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)



    #complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

print(train.isnull().sum())

print("-"*10)

print(test.isnull().sum())
train['AgeGroup'] = pd.cut(train['Age'].astype(int), 5)

train[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean().sort_values(by='AgeGroup', ascending=True)
# replacing age with ordinal values in these groups

for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ (dataset['Age'] > 64) & (dataset['Age'] <= 80),'Age'] = 4

train.head()
train['FareGroup'] = pd.cut(train['Fare'],4)

train[['FareGroup', 'Survived']].groupby(['FareGroup'], as_index=False).mean().sort_values(by='FareGroup', ascending=True)
# replacing fare with ordinal values in these groups

for dataset in combine:    

    dataset.loc[ dataset['Fare'] <= 128, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 128) & (dataset['Fare'] <= 256), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 256) & (dataset['Fare'] <= 384), 'Fare'] = 2

    dataset.loc[(dataset['Fare'] > 384), 'Fare'] = 3

combine = [train, test]

train.head()
for dataset in combine:    

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
for dataset in combine:

    dataset['IsAlone'] = 1 #initialize to yes/1 is alone

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 
for dataset in combine:

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

train.head()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

for dataset in combine:

    dataset['Sex'] = labelencoder.fit_transform(dataset['Sex'])
for dataset in combine:

    dataset['Embarked'] = labelencoder.fit_transform(dataset['Embarked'])

for dataset in combine:

    dataset['Title'] = labelencoder.fit_transform(dataset['Title'])
train.head()
test.head()
print('Before', train.shape, test.shape, combine[0].shape, combine[1].shape)



train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'AgeGroup', 'FareGroup' ], axis = 1)

test = test.drop(['Name', 'Ticket', 'Cabin'], axis = 1)

combine = [train, test]



print('After', train.shape, test.shape, combine[0].shape, combine[1].shape)
train.head(5)
#MinMaxScaled

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1,1))

for dataset in combine:

    dataset['Age'] = scaler.fit_transform(dataset.Age.values.reshape(-1, 1))
train.head()
#MinMaxScaled

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1,1))

for dataset in combine:

    dataset['Fare'] = scaler.fit_transform(dataset.Age.values.reshape(-1, 1))
train.head()
X_train = train.drop('Survived', axis = 1).copy()

Y_train = train['Survived']

X_test = test.drop('PassengerId', axis = 1).copy()

X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

accu_reg = round(logreg.score(X_train, Y_train) * 100, 2)

accu_reg
svc = SVC(kernel = 'rbf', C =1, gamma = 0.1)

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

accu_svc = round(svc.score(X_train, Y_train) *100, 2)

accu_svc
svc = SVC(kernel = 'linear', C =1, gamma = 0.1)

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

accu_lsvc = round(svc.score(X_train, Y_train) *100, 2)

accu_lsvc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

accu_knn = round(svc.score(X_train, Y_train) *100, 2)

accu_knn
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
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(random_state=0)

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

clf.score(X_train, Y_train)

acc_clf = round(clf.score(X_train, Y_train)*100, 2)

acc_clf
models = pd.DataFrame({

    'Model': ['Radial Support Vector Machines', 'Linear Support Vector Machine', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree', 'Gradient Boosting Classifier'],

    'Score': [accu_svc ,accu_lsvc, accu_knn, accu_reg, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree, acc_clf]})

models.sort_values(by='Score', ascending=False)
from sklearn.model_selection import KFold #for K-fold cross validation

#from sklearn.model_selection import cross_val_score #score evaluation

#from sklearn.model_selection import cross_val_predict #prediction

kfold = KFold(n_splits=10) # k=10, split the data into 10 equal parts

kfold.get_n_splits(X_test)

from sklearn.ensemble import AdaBoostClassifier

ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)

result=cross_val_score(ada,X_train,Y_train,cv=10,scoring='accuracy')

print('The cross validated score for AdaBoost is:',result.mean())
from sklearn.ensemble import GradientBoostingClassifier

grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)

result=cross_val_score(grad,X_train,Y_train,cv=10,scoring='accuracy')

print('The cross validated score for Gradient Boosting is:',result.mean())
import xgboost as xg

xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)

result=cross_val_score(xgboost,X_train,Y_train,cv=10,scoring='accuracy')

print('The cross validated score for XGBoost is:',result.mean())
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })



submission.head(20)

submission.to_csv('Submission.csv', index=False)
sample_submission.head(20)
g=sns.FacetGrid(sample_submission, col = 'Survived')

g.map(plt.hist,'PassengerId', bins = 20)
g=sns.FacetGrid(submission, col = 'Survived')

g.map(plt.hist,'PassengerId', bins = 20)
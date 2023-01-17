# data analysis and wrangling

import pandas as pd

import numpy as np

from pandas_profiling import ProfileReport

# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.tree import DecisionTreeClassifier



import warnings

warnings.filterwarnings('ignore')



#check version

import sys

print(sys.version)
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head(10)
test.head()
train.shape
train.describe()
train.isnull().sum()
test.isnull().sum()
train.describe(percentiles=[0.05,.1,.2,.3,0.4,0.5,0.6,0.7,0.8,0.9,0.98])
test.describe(percentiles=[0.05,.1,.2,.3,0.4,0.5,0.6,0.7,0.8,0.9,0.98])
train.corr().style.background_gradient(cmap='Blues')
train_test = [train,test]

train.head()
print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
sex_mapping = {"male": 1, "female": 0}

for dataset in train_test:

    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
for dataset in train_test:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in train_test:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
for dataset in train_test:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
for dataset in train_test:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
for dataset in train_test:

    age_avg    = dataset['Age'].mean()

    age_std    = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

print (train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean())
for df in train_test:

    df['Person']=df['Name'].str.extract('([A-Za-z]*)\.')

train['Person'].value_counts()
test['Person'].value_counts()
title_mapping = { "Mr": 1, "Miss": 2, "Mrs": 3,"Master": 4, "Dr": 5, "Rev": 5, "Col": 5, "Major": 5, "Mlle": 5,"Countess": 5,

                 "Ms": 5, "Lady": 5, "Jonkheer": 5, "Don": 5, "Dona" : 5, "Mme": 5,"Capt": 5,"Sir": 5 }

for dataFrame in train_test:

    dataFrame['Person']=dataFrame['Person'].map(title_mapping)
print(pd.crosstab(train['Person'], train['Sex']))
for dataset in train_test:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# Mapping Fare

    

#for dataset in train_test:

 #   dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']   = 0

  #  dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

   # dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

 #   dataset.loc[ dataset['Fare'] > 31, 'Fare']      = 3

  #  dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age']     = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] =4
train
# Feature Selection

drop_list = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',\

                 'Parch','Fare','FamilySize']

train = train.drop(drop_list, axis = 1)

test  = test.drop(['Name', 'Ticket', 'Cabin', 'SibSp',\

                 'Parch', 'Fare', 'FamilySize'], axis = 1)



train.head(10)
test.head(10)
# Importing Classifier Modules

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression





from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis



classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]

X_train=train.drop('Survived',axis=1)

Y_train=train['Survived']



X_test = test.drop('PassengerId',axis=1).copy()

# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
from sklearn.model_selection import KFold , cross_val_score



#10 splits of Kfolds 

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)



scoring = 'accuracy'

score = cross_val_score(svc, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)

score
#LogisticRegression Model

Lr = LogisticRegression()

Lr.fit(X_train, Y_train)

prediction = Lr.predict(X_test)

acc_lr = round(Lr.score(X_train, Y_train) * 100, 2)

acc_lr
#GaussianNaifBayesien

Gs = GaussianNB()

Gs.fit(X_train, Y_train)

prediction = Gs.predict(X_test)

acc_gs = round(Gs.score(X_train, Y_train) * 100, 2)

acc_gs
#KNN Model

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



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
models = pd.DataFrame({

    'Model': ['Random Forest', 'Support Vector Machines', 'Logistic Regression', 

              'Gaussian Naif Bayesien', 'KNeighbors Classifier',

              'Linear SVC', 'Stochastic Gradient Descent', 

              'Decision Tree'],

    'Score': [acc_random_forest, acc_svc, 

              acc_lr, 

              acc_gs, acc_knn, acc_linear_svc ,  acc_sgd ,  acc_decision_tree       ]})

models.sort_values(by='Score', ascending=False)
#Y_pred = random_forest.predict(X_test)

#Y_pred = knn.predict(X_test)

#Y_pred = decision_tree.predict(X_test)
Y_pred = decision_tree.predict(pd.get_dummies(X_test))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })



submission.to_csv('submission1.csv', index=False)
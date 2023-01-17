# iports that we need:

# plotting:

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()



# data analysis

import numpy as np

import pandas as pd



#ignore warnings:

import warnings

warnings.filterwarnings('ignore')



# so we can import data

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# load the data

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
# first look:

train.describe(include='all')
fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(train.corr(), annot=True, linewidth=0.5, fmt='.2f', ax=ax)

ax.set_ylim(7,0);
def make_plot(x, y='Survived', df=train):

    sns.barplot(x=x, y=y, data=df)
make_plot('Sex')
make_plot('Pclass')
make_plot('Parch')
make_plot('SibSp')
# making the bins:

bins = [-np.inf, 0, 5, 12, 18, 25, 35, 55, np.inf]

labels = ['unknown', 'infant', 'child', 'teenager', 'student', 'young_adult', 'adult', 'old']



for data in [train, test]: # note I do this to the test set too

    data['AgeGrp'] = pd.cut(data['Age'], bins=bins, labels=labels)
fig, ax = plt.subplots(figsize=(10,4))

sns.barplot(x="AgeGrp", y='Survived', hue="Sex", data=train, ax=ax);

# survival accounting for gender
sns.barplot(x="AgeGrp", y='Survived', data=train);

# total survival
pd.DataFrame(train.isnull().sum(), columns=['Train']).join(

                pd.DataFrame(test.isnull().sum(), columns=['Test'])

                )
test[test['Fare'].isnull()]
median_fare = test.groupby(['Pclass']).Fare.median()[3] # as he was in 3rd class

test['Fare'].fillna(median_fare, inplace=True)
train[train['Embarked'].isnull()]
sns.countplot(x='Embarked', data=train);
train['Embarked'] = train['Embarked'].fillna('S')
train['Age'] = train.groupby(['Pclass', 'SibSp'])['Age'].apply(lambda x: x.fillna(x.median()))

train[train['Age'].isnull()]
train['Age'].fillna(11, inplace=True)

train['Age'].isnull().sum()
test['Age'] = test.groupby(['Pclass', 'SibSp'])['Age'].apply(

    lambda x: x.fillna(x.median()))

test['Age'].isnull().sum()
for data in [train, test]:

    for feature in ['PassengerId', 'Cabin', 'AgeGrp']:

        data.drop(feature, inplace=True, axis=1)
train.head()
for dataset in [train, test]:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



# code from https://www.kaggle.com/startupsci/titanic-data-science-solutions    



pd.crosstab(train['Title'], train['Sex'])
# grouping the uncommon names:

train['Title'] = train['Title'].replace(['Ms', 'Mlle'],'Miss')

train['Title'] = train['Title'].replace(['Mme'],'Mrs')

train['Title'] = train['Title'].replace(['Dr','Rev','the','Jonkheer','Lady','Sir', 'Don', 'Countess'],'Nobles')

train['Title'] = train['Title'].replace(['Major','Col', 'Capt'],'Navy')

train.Title.value_counts()
sns.barplot(x = 'Title', y = 'Survived', data=train);
# and for the tesst data - not all are present:



test['Title'] = test['Title'].replace(['Ms','Dona'],'Miss')

test['Title'] = test['Title'].replace(['Dr','Rev'],'Nobles')

test['Title'] = test['Title'].replace(['Col'],'Navy')

test.Title.value_counts()
categorical_features = [ 'Sex', 'Title', 'Embarked']



for feature in categorical_features:

    dummies = pd.get_dummies(train[feature]).add_prefix(feature+'_')

    train = train.join(dummies)



for feature in categorical_features:

    dummies = pd.get_dummies(test[feature]).add_prefix(feature+'_')

    test = test.join(dummies)
for data in [train, test]:

    for feature in ['Name', 'Sex', 'Title', 'Embarked', 'Ticket']:

        data.drop(feature, axis=1, inplace=True)
# independant and dependant variables:

X = train.drop('Survived', axis=1)

y = train['Survived']
# this will split the model when we want to check our model.

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.22, random_state = 0)
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_val = sc.transform(X_val)

test = sc.transform(test)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB



gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

y_pred = gaussian.predict(X_val)

round(accuracy_score(y_pred, y_val) * 100, 2)
# decision tree:

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, KFold





#Here we will use gridsearchcv to find the best values for our hyperparameter

# kfold is for its internal validation:

cv = KFold(n_splits=10, shuffle=True, random_state=42)



params = dict(max_depth=range(1,10),

              max_features=[2, 4, 6, 8],

              criterion=['entropy', 'gini']

             )

DTGrid = GridSearchCV(DecisionTreeClassifier(random_state=42),

                    param_grid=params, verbose=False,

                    cv=cv)



DTGrid.fit(X_train, y_train)

DecTree = DTGrid.best_estimator_

print(DTGrid.best_params_)

round(DecTree.score(X_val, y_val) * 100, 2)
# random forest, using grid search as above:

from sklearn.ensemble import RandomForestClassifier



cv=KFold(n_splits=10, shuffle=True, random_state=42)



params = {'n_estimators': [80, 100, 120, 140],

              'max_depth': range(2,7),

              'criterion': ['gini', 'entropy']      

        }





RFGrid = GridSearchCV(RandomForestClassifier(random_state=42),

                    param_grid=params, verbose=False,

                    cv=cv)



RFGrid.fit(X_train, y_train)

RandForest = RFGrid.best_estimator_

print(RFGrid.best_params_)

round(RandForest.score(X_val, y_val) * 100, 2)
# k nearest neighbors:

from sklearn.neighbors import KNeighborsClassifier



params = dict(n_neighbors=[3,6,8,10],

              weights=['uniform', 'distance'],

              metric=['euclidean', 'manhattan']

              )

cv=KFold(n_splits=10, shuffle=True, random_state=42)



KNNGrid = GridSearchCV(KNeighborsClassifier(),

                    param_grid=params, verbose=False,

                    cv=cv)



KNNGrid.fit(X_train, y_train)

KNN = KNNGrid.best_estimator_

print(KNNGrid.best_params_)

round(KNN.score(X_val, y_val) * 100, 2)
# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

logreg.score(X_val, y_val)
# Support Vector Machines:

from sklearn.svm import SVC



params = {'C':[100,500,1000],

          'gamma':[0.1,0.001,0.0001],

          'kernel':['linear','rbf']

          }

cv=KFold(n_splits=10, shuffle=True, random_state=42)



SVMGrid = GridSearchCV(SVC(random_state=42),

                    param_grid=params, verbose=False,

                    cv=cv)



SVMGrid.fit(X_train, y_train)

SVM = SVMGrid.best_estimator_

print(SVMGrid.best_params_)

round(SVM.score(X_val, y_val) * 100, 2)
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



params = {'loss':['hinge', 'perceptron'],

          'alpha':[0.01, 0.001, 0.0001],

          'penalty':['l2', 'l1']

          }

cv=KFold(n_splits=10, shuffle=True, random_state=42)



SGDGrid = GridSearchCV(SGDClassifier(random_state=42),

                    param_grid=params, verbose=False,

                    cv=cv)



SGDGrid.fit(X_train, y_train)

SGD = SGDGrid.best_estimator_

print(SGDGrid.best_params_)

round(SGD.score(X_val, y_val) * 100, 2)
# gradient Boosting:

from sklearn.ensemble import GradientBoostingClassifier



params = {'loss':[ 'deviance', 'exponential'],

          'learning_rate':[ 0.1, 0.01, 0.001],

          'n_estimators':[100, 400, 700]

          }



cv=KFold(n_splits=10, shuffle=True, random_state=42)



GBGrid = GridSearchCV(GradientBoostingClassifier(random_state=42),

                    param_grid=params, verbose=False,

                    cv=cv)



GBGrid.fit(X_train, y_train)

GB = GBGrid.best_estimator_

print(GBGrid.best_params_)

round(GB.score(X_val, y_val) * 100, 2)
model = SGDClassifier(**SGDGrid.best_params_)

model.fit(X, y)



test_id = pd.read_csv('../input/titanic/test.csv')

submission = pd.DataFrame({'PassengerId': test_id['PassengerId'], 'Survived': model.predict(test) })

submission.PassengerId = submission.PassengerId.astype(int)

submission.Survived = submission.Survived.astype(int)

submission.to_csv('submission.csv', index=False)
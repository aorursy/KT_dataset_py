# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Modelling Algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier



from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve





# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.cross_validation import train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV



from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle



# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# get titanic & test csv files as a DataFrame

train = pd.read_csv("../input/train.csv")

test    = pd.read_csv("../input/test.csv")

print (train.shape, test.shape)
train.head()
train.describe()
#Checking for missing data

NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])

NAs[NAs.sum(axis=1) > 0]
# Spliting to features and lables

train_labels = train.pop('Survived')



features = pd.concat([train, test], keys=['train', 'test'])

features.shape
# At this point we will drop the Cabin feature since it is missing a lot of the data

features.pop('Cabin')



# At this point names don't affect our model so we drop it

#features.pop('Name')



# At this point we drop Ticket feature

features.pop('Ticket')



features.shape


features['Name'].head()

i = features['Name'][0]

i.split(',')[1].split('.')[0].strip()
# Get Title from Name

dataset_title = [i.split(',')[1].split('.')[0].strip() for i in features['Name']]

features['Title'] = dataset_title

features['Title'].head()
g = sns.countplot(x="Title",data=features)

g = plt.setp(g.get_xticklabels(), rotation=45)
# Convert to categorical values Title

features["Title"] = features["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

features["Title"] = features["Title"].map({"Master":'0', "Miss":'1', "Ms":'1', "Mme":'1', "Mlle":'1', "Mrs":'1', "Mr":'2', "Rare":'3'})
g = sns.countplot(features["Title"])

g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
features.pop('Name')

features.shape
# Filling missing Age values with mean

features['Age'] = features['Age'].fillna(features['Age'].mean())



# Filling missing Embarked values with most common value

features['Embarked'] = features['Embarked'].fillna(features['Embarked'].mode()[0])



# Filling missing Fare values with mean

features['Fare'] = features['Fare'].fillna(features['Fare'].mean())
#Checking for missing data

NAs = pd.concat([features.isnull().sum()], axis=1, keys=['features'])

NAs[NAs.sum(axis=1) > 0]
features.shape
# Apply log to Fare to reduce skewness distribution

features["Fare"] = features["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
def cat2num(x):

    if x == 'male':

        return 1

    else:

        return 0
features['Sex'] = features['Sex'].apply(cat2num)
def num2cat(x):

    return str(x)
features['Pclass_num'] = features['Pclass'].apply(num2cat)

features.pop('Pclass')

features.shape
features['Family'] = features['SibSp'] + features['Parch'] + 1

features.pop('SibSp')

features.pop('Parch')

features.shape
# Getting Dummies from all other categorical vars

for col in features.dtypes[features.dtypes == 'object'].index:

    for_dummy = features.pop(col)

    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)
features.shape
features.head()
### Splitting features

train_features = features.loc['train'].drop('PassengerId', axis=1).select_dtypes(include=[np.number]).values

test_features = features.loc['test'].drop('PassengerId', axis=1).select_dtypes(include=[np.number]).values
## META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING



# Adaboost

DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsadaDTC = GridSearchCV(adaDTC, param_grid = ada_param_grid, cv=5,

                        scoring="accuracy", n_jobs= -1, verbose = 1)



gsadaDTC.fit(train_features, train_labels)



ada_best = gsadaDTC.best_estimator_

gsadaDTC.best_score_
#ExtraTrees 

ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=5,

                      scoring="accuracy", n_jobs= -1, verbose = 1)



gsExtC.fit(train_features, train_labels)



ExtC_best = gsExtC.best_estimator_



# Best score

gsExtC.best_score_
# RFC Parameters tunning 

RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=5,

                     scoring="accuracy", n_jobs= -1, verbose = 1)



gsRFC.fit(train_features, train_labels)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
# Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=5,

                     scoring="accuracy", n_jobs= -1, verbose = 1)



gsGBC.fit(train_features, train_labels)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
### SVC classifier

SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=5,

                      scoring="accuracy", n_jobs= -1, verbose = 1)



gsSVMC.fit(train_features, train_labels)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('gbr', GBC_best),

                                       ('svc', SVMC_best), ('ada', ada_best),

                                       ('Ext', ExtC_best)],

                           n_jobs=-1)



votingC = votingC.fit(train_features, train_labels)
test_y = votingC.predict(test_features)
test_id = test.PassengerId

test_submit = pd.DataFrame({'PassengerId': test_id, 'Survived': test_y})

test_submit.shape

test_submit.head()

#test_submit.to_csv('titanic_voting.csv', index=False)
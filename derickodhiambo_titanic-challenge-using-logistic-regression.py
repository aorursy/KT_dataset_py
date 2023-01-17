#importing necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import math

import seaborn as sns

sns.set()

import os

#loading data

test = pd.read_csv('../input/titanic/test.csv')

train = pd.read_csv('../input/titanic/train.csv')
train.shape
#Display top 3 rows

train.head(3)
# Getting information of non-null objects

train.info()
train.isnull().sum()
round(train.describe(include = 'all', percentiles=[0.01,0.1,0.25,0.75,0.99]))
#STATISTICS

## Mean: Average value of a column

train.Fare.sum()/len(train.Fare)

np.mean(train.Fare)

train.Fare.mean()

# Median: middle value after sorting the column

train.Fare.median()
# Mode: Value that appears most frequently in the data

train.Fare.mode()

train.Fare.value_counts()
train.Fare.std()
train.Fare.var()
train.columns
train.Pclass.value_counts()
# distribution

train.Fare.plot.hist()
#Analyzing the data

sns.countplot(x="Survived", data=train)
#checking survival rate of each Sex

sns.countplot(x="Survived", hue="Sex", data=train)
#checking survival rate of passengers from each class

sns.countplot(x="Survived", hue="Pclass", data=train)
#clasification

train.info()
train.Age.isnull().sum()
train.head()
#Data cleaning

# Drop unnecessary columns or columns with lot of missing values

train = train.drop(['PassengerId','Name','Ticket','Embarked','Cabin'], axis=1)

train.head()
#Data cleaning

# Replace missing value in age with

train['Age'] = train['Age'].fillna(train['Age'].mean())
train.info()
train.head()
# Change sex variable to have numeric values

train['Sex'].replace(['male','female'],[0,1],inplace=True)
train.head()
# importing the libraries to split the data and Splitting the data

from sklearn.model_selection import train_test_split

#training and testing data

train,test = train_test_split(train, test_size=0.3, random_state=0, stratify=train['Survived'])
train.shape
test.shape
train.head(3)
train.columns
# separating dependent and independent variables

train_X=train[train.columns[1:]]

train_Y=train[train.columns[:1]]

test_X=test[test.columns[1:]]

test_Y=test[test.columns[:1]]
train_X.head()
train_Y.head()
# Importing and applying logistic regression model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(train_X,train_Y)
test_X.head()
predictions=model.predict(test_X)
test_X.shape
predictions
# Importing libraries for evaluation metrics

from sklearn import metrics

from sklearn.metrics import confusion_matrix
test_Y.head()
metrics.accuracy_score(test_Y,predictions)
metrics.precision_score(test_Y,predictions)
metrics.recall_score(test_Y,predictions)
pd.DataFrame(confusion_matrix(test_Y,predictions),\

columns=["Predicted No", "Predicted Yes"],\

index=["Actual No","Actual yes"] )
#Accuracy

(149+72)/(149+16+31+72)
# Precision

72/(16+72)
# Recall

72/(31+72)
#AUC-ROC (Area Under Curve - Receiver Operating Characteristic)

# Getting prediction probabilities

probs = model.predict_proba(test_X)[:, 1]
# Calculating ROC_AUC

from sklearn.metrics import roc_curve, auc

FPR, TPR, thresholds = roc_curve(test_Y, probs)

ROC_AUC = auc(FPR, TPR)

ROC_AUC
# Plotting ROC_AUC

import matplotlib.pyplot as plt

plt.plot(FPR,TPR)

plt.xlabel('False Positive Rate: FP/(FP+TN)', fontsize = 18)

plt.ylabel('True Positive Rate (Recall)', fontsize = 18)

plt.title('ROC for survivors', fontsize= 18)

plt.show()
#Cross-Validation

# Applying cross validation

# StratifiedShuffleSplit: stratified randomized folds

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25, random_state = 0 )
import warnings

warnings.filterwarnings("ignore")
test.head()
X = train[train.columns[1:]]

y = train[train.columns[:1]]
## saving the feature names for decision tree display

column_names = X.columns

# Scaling the data

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
X
accuracies = cross_val_score(LogisticRegression(solver='liblinear'), X, y, cv = cv)

print ("Cross-Validation accuracy scores:{}".format(accuracies))

print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),5)))
#Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV, StratifiedKFold

C_vals = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,16.5,17,17.5,18]

penalties = ['l1','l2']

cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25)

param = {'penalty': penalties, 'C': C_vals}

grid = GridSearchCV(estimator=LogisticRegression(),

param_grid = param,

scoring = 'accuracy',

n_jobs = -1,

cv = cv

)

grid.fit(X, y)
## Getting the best of everything.

print(grid.best_score_)

print(grid.best_params_)

print(grid.best_estimator_)
lr_grid = grid.best_estimator_

lr_grid.score(X,y)
#KNN

## Importing the model.

from sklearn.neighbors import KNeighborsClassifier

## calling on the model oject.

knn = KNeighborsClassifier(metric='minkowski', p=2)

## doing 10 fold staratified-shuffle-split cross validation

cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=2)

accuracies = cross_val_score(knn, X,y, cv = cv, scoring='accuracy')

print ("Cross-Validation accuracy scores:{}".format(accuracies))

print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),3)))
## Search for an optimal value of k for KNN.

k_range = range(1,31)

k_scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X,y, cv = cv, scoring = 'accuracy')

    k_scores.append(scores.mean())

print("Accuracy scores are: {}\n".format(k_scores))

print ("Mean accuracy score: {}".format(np.mean(k_scores)))
from matplotlib import pyplot as plt

plt.plot(k_range, k_scores, c = 'green')
from sklearn.model_selection import GridSearchCV

## trying out multiple values for k

k_range = range(1,31)

weights_options=['uniform','distance']

param = {'n_neighbors':k_range, 'weights':weights_options}

## Using startifiedShufflesplit.

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

# estimator = knn, param_grid = param, n_jobs = -1 to instruct scikit learn to use all available process

grid = GridSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1)

## Fitting the model.

grid.fit(X,y)
print(grid.best_score_)

print(grid.best_params_)

print(grid.best_estimator_)
### Using the best parameters from the grid-search.

knn_grid= grid.best_estimator_

knn_grid.score(X,y)
#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

gaussian = GaussianNB()

gaussian.fit(train_X, train_Y)

predictions = gaussian.predict(test_X)

gaussian_accy = round(accuracy_score(test_Y, predictions), 3)

print(gaussian_accy)
#SVM

from sklearn.svm import SVC

Cs = [0.001, 0.01, 0.1, 1,1.5,2,2.5,3,4,5, 10] ## penalty parameter C for the error term

gammas = [0.0001,0.001, 0.01, 0.1, 1]

param_grid = {'C': Cs, 'gamma' : gammas}

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

## 'rbf' stands for gaussian kernel

grid_search = GridSearchCV(SVC(kernel = 'rbf', probability=True), param_grid, cv=cv)

grid_search.fit(X,y)
print(grid_search.best_score_)

print(grid_search.best_params_)

print(grid_search.best_estimator_)
# using the best found hyper paremeters to get the score.

svm_grid = grid_search.best_estimator_

svm_grid.score(X,y)
#Decision Trees

from sklearn.tree import DecisionTreeClassifier

max_depth = range(1,20)

max_feature = [20,22,24,26,28,30,'auto']

criterion=["entropy", "gini"]



param = {'max_depth':max_depth,

         'max_features':max_feature,

         'criterion': criterion

        }

grid = GridSearchCV(DecisionTreeClassifier(),

                    param_grid = param,

                     verbose=False,

                     cv=StratifiedKFold(n_splits=5, random_state=15, shuffle=True),

                    n_jobs = -1)

grid.fit(X, y)
print( grid.best_params_)

print (grid.best_score_)

print (grid.best_estimator_)
dectree_grid = grid.best_estimator_

dectree_grid.score(X,y)
#Bagging

#Random Forest

from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

n_estimators = [140,145,150,155,160];

max_depth = range(1,10);

criterions = ['gini', 'entropy'];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)



parameters = {'n_estimators':n_estimators,

              'max_depth':max_depth,

              'criterion': criterions}

grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),

                    param_grid=parameters,

                    cv=cv,

                    n_jobs = -1)

grid.fit(X,y)
print (grid.best_score_)

print (grid.best_params_)

print (grid.best_estimator_)
rf_grid = grid.best_estimator_

rf_grid.score(X,y)
#feature importance

feature_importances = pd.DataFrame(rf_grid.feature_importances_,

index = test[test.columns[1:]].columns,

columns=['importance'])

feature_importances.sort_values(by='importance', ascending=False)
#Boosting

#GBM

from sklearn.ensemble import GradientBoostingClassifier

n_estimators = [100,140,145,150,160, 170,175,180,185];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

learning_r = [0.1,1,0.01,0.5]



parameters = {'n_estimators':n_estimators,

              'learning_rate':learning_r}

grid = GridSearchCV(GradientBoostingClassifier(),

                    param_grid=parameters,

                    cv=cv,

                    n_jobs = -1)

grid.fit(X,y)
print (grid.best_score_)

print (grid.best_params_)

print (grid.best_estimator_)
GBM_grid = grid.best_estimator_

GBM_grid.score(X,y)
from xgboost import XGBClassifier

n_estimators = [100,140,145,150,200,400];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

learning_r = [0.1,1,0.01,0.5]



parameters = {'n_estimators':n_estimators,

              'learning_rate':learning_r

             }

grid = GridSearchCV(XGBClassifier(),

                    param_grid=parameters,

                    cv=cv,

                    n_jobs = -1)

grid.fit(X,y)
print (grid.best_score_)

print (grid.best_params_)

print (grid.best_estimator_)
XGB_grid = grid.best_estimator_

XGB_grid.score(X,y)
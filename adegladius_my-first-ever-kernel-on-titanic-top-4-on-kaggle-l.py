# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading of dependencies

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost

import csv as csv

from xgboost import plot_importance

from matplotlib import pyplot

from sklearn.model_selection import cross_val_score,KFold

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

from scipy.stats import skew

from collections import OrderedDict

from numpy import loadtxt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
import pandas as pd

Titanic_test = pd.read_csv('/kaggle/input/Test.csv')

Titanic_train = pd.read_csv('/kaggle/input/Train.csv')
Titanic_train.head()
Titanic_test.head()
Data_value = Titanic_train.drop(['PassengerId', 'Survived'], axis=1)

Data_target = Titanic_train['Survived']
## Splitting of traindata into two

X_train, X_test, y_train, y_test = train_test_split(Data_value, Data_target,random_state = 21,test_size = 0.30)

           
X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_test.head()
# Building of XGBoost Classifier

model = XGBClassifier(n_estimators= 89,gamma=0.1,eta = 0.05,

              learning_rate=0.01,colsample_bytree=0.8, max_delta_step=0, max_depth=4,

              min_child_weight=1, missing=None, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state= 123,

              reg_alpha=0.75, reg_lambda=0.45, scale_pos_weight=1, seed= 42,

              silent=None, subsample= 0.8)



eval_set = [(X_train, y_train), (X_test, y_test)]

model.fit(X_train, y_train.values.ravel(), early_stopping_rounds=10, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
# Checking Train accuracy

accuracy = model.score(X_train, y_train)

accuracy

print("accuracy: %.1f%%" % (accuracy * 100.0))
## Evaluation of Test

X_test.head()
# make predictions for test data through X_data set above

y_pred = model.predict(X_test)

#predictions = [round(value) for value in y_pred]

print(y_pred)
# Print classification report

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
# Prediction accuracy

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.1f%%" % (accuracy * 100))
# retrieve performance metrics

results = model.evals_result()

epochs = len(results['validation_0']['error'])

x_axis = range(0, epochs)



# plot log loss

fig, ax = pyplot.subplots()

ax.plot(x_axis, results['validation_0']['logloss'], label='Train')

ax.plot(x_axis, results['validation_1']['logloss'], label='Test')

ax.legend()

pyplot.ylabel('Log Loss')

pyplot.title('XGBoost Log Loss')

pyplot.show()



# plot classification error

fig, ax = pyplot.subplots()

ax.plot(x_axis, results['validation_0']['error'], label='Train')

ax.plot(x_axis, results['validation_1']['error'], label='Test')

ax.legend()

pyplot.ylabel('Classification Error')

pyplot.title('XGBoost Classification Error')

pyplot.show()
# Confusion Matrix

from sklearn.metrics import confusion_matrix

conf = confusion_matrix(y_test,y_pred)

conf
## Make a prediction on test datset 

sub_test = Titanic_test.drop(['PassengerId'], axis =1)

sub_test_pred = model.predict(sub_test).astype(int)
# Make a submission to Kaggle

Allsub = pd.DataFrame({'PassengerId':Titanic_test['PassengerId'],

                       'Survived' :sub_test_pred})

Allsub.to_csv("Submission_Titanic_XGBoost.csv", index = False)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

X = Titanic_train.iloc[:,2:] # Independent Columns

y = Titanic_train.iloc[:,1] # target column i.e Survived

Titanic_train.head(1)
# Apply SelectKBest class to extract top best features

bestfeatures = SelectKBest(score_func=chi2, k= 'all')

fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)
# Concatenate two dataframes for better visualisation

featureScores = pd.concat([dfcolumns,dfscores], axis =1)

# Naming of dataframe columns

featureScores.columns = ['Independent variables', 'Score']
featureScores
# Print 11 best features

print(featureScores.nlargest(11,'Score'))
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model2 = ExtraTreesClassifier()

model2.fit(X,y)
print(model2.feature_importances_)
## Plot Graph of feature importances for better visualization

feat_importances = pd.Series(model2.feature_importances_, index = X.columns) 

feat_importances.nlargest(11).plot(kind = 'barh')

plt.show()
## Checking features correlation 

## (using Correaltion Matrix with Heatmap)

import seaborn as sns

## To extract the best 11 features from our train data

# so that we can plot only those 11 best features using HeatMap

# according to their column number in our train data set.

X1 = Titanic_train.iloc[:,9:19]

corrmat = X1.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(10000,10000))
# Plot Heat-Map

g = sns.heatmap(X1[top_corr_features].corr(),

                annot=True,cmap='RdYlGn')
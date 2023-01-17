# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
credit = pd.read_csv('/kaggle/input/creditcarddefault/credit-card-default.csv')

credit.head()
credit.info()
X = credit.drop('defaulted', axis=1)

Y = credit['defaulted']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=101)
model = RandomForestClassifier()

model.fit(X_train, Y_train)
predicted = model.predict(X_test)
print('Accuracy', round(accuracy_score(predicted, Y_test)*100))
n_folds = 5

parameters = {'max_depth':range(2,20,5)}
rfModel = RandomForestClassifier()

rfModel = GridSearchCV(rfModel, parameters, cv=n_folds, scoring='accuracy')
rfModel.fit(X_train, Y_train)
scores = rfModel.cv_results_

pd.DataFrame(scores).head()
# GridSearchCV to find optimal n_estimators

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'n_estimators': range(100, 1500, 400)}



# instantiate the model (note we are specifying a max_depth)

rf = RandomForestClassifier(max_depth=4)





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_train, Y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# GridSearchCV to find optimal max_features

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_features': [4, 8, 14, 20, 24]}



# instantiate the model

rf = RandomForestClassifier(max_depth=4)





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_train, Y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# GridSearchCV to find optimal min_samples_leaf

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_leaf': range(100, 400, 50)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_train, Y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# GridSearchCV to find optimal min_samples_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_split': range(200, 500, 50)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_train, Y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [4,8,10],

    'min_samples_leaf': range(100, 400, 200),

    'min_samples_split': range(200, 500, 200),

    'n_estimators': [100,200, 300], 

    'max_features': [5, 10]

}

# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1,verbose = 1)
# Fit the grid search to the data

grid_search.fit(X_train, Y_train)
# printing the optimal accuracy score and hyperparameters

print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)
# model with the best hyperparameters

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(bootstrap=True,

                             max_depth=10,

                             min_samples_leaf=100, 

                             min_samples_split=200,

                             max_features=10,

                             n_estimators=100)
# fit

rfc.fit(X_train,Y_train)
# predict

predictions = rfc.predict(X_test)
# evaluation metrics

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
((6752+689)/(6752+306+1253+689))
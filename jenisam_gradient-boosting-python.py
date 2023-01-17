# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt

from sklearn.ensemble import GradientBoostingClassifier #GBM algorithm
from sklearn import cross_validation, metrics #Additional scklearn functions
from sklearn.grid_search import GridSearchCV #performing grid search

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/diabetes.csv")
#build a quick logistic regression model and check the accuracy

#X = df.iloc[:,:8] # independent variables
y = 'Class' # dependent variables

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Class'])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Class'], 
                                                    cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Class'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Class'], dtrain_predprob))
    
    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
#Choose all predictors except target
predictors = df.columns.values[:8]
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, df, predictors)
predictors = df.columns.values[:8]
gbm01 = GradientBoostingClassifier(learning_rate=0.1,
                                   n_estimators=60,
                                   max_depth=9,
                                   subsample=0.8,
                                   random_state=10)
modelfit(gbm01, df, predictors)
import pandas as pd
import numpy as np

# Bagged Decision Trees for Classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from sklearn.model_selection import train_test_split

import os

print(os.listdir("../input"))
df = pd.read_csv("../input/diabetes.csv")
#build a quick logistic regression model and check the accuracy

X = df.iloc[:,:8] # independent variables
y = df['Class'] # dependent variables
#Normalize
X = preprocessing.StandardScaler().fit_transform(X)
# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
import sklearn.cross_validation as cross_validation
kfold = cross_validation.StratifiedKFold(y = y_train, n_folds=5, random_state=2017)
num_trees = 100
# Dection Tree with 5 fold cross validation
# lets restrict max_depth to 3 to have more impure leaves
clf_DT = DecisionTreeClassifier(max_depth=1, random_state=2017).fit(X_train,y_train)
results = cross_validation.cross_val_score(clf_DT, X_train,y_train,cv=kfold)
print ("Decision Tree (stand alone) - Train : ", results.mean())
print ("Decision Tree (stand alone) - Test : ", metrics.accuracy_score(clf_DT.predict(X_test), y_test))
# Using Adaptive Boosting of 100 iteration
clf_DT_Boost = AdaBoostClassifier(base_estimator=clf_DT, n_estimators=num_trees, 
                                  learning_rate=0.1, random_state=2017).fit(X_train,y_train)
results = cross_validation.cross_val_score(clf_DT_Boost, X_train, y_train,
cv=kfold)
print ("Decision Tree (AdaBoosting) - Train : ", results.mean())
print ("Decision Tree (AdaBoosting) - Test : ", metrics.accuracy_score(clf_DT_Boost.predict(X_test), y_test))

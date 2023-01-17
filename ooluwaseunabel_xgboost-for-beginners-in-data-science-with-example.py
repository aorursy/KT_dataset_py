# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from numpy import loadtxt



#Import XGBoost Model

from xgboost import XGBClassifier

from xgboost import plot_importance

from matplotlib import pyplot

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

#Success

print ('Run Successful')

# Any results you write to the current directory are saved as output.
#Import the Pina Indians Diabetes Dataset

dataset = loadtxt("../input/Diabetes.csv" , delimiter = ",")

print ("Run Successfully")
#Split the Dataset into X and Y

X = dataset[:, 0:8]

Y = dataset [:,8]

print ('Ran Successfully')
#Split the Dataset into into Train and Test 

seed = 7

test_size = 0.33

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size, random_state = seed)

print ('Ran Successfully')
#Let's fit our model on the training data

xgb = XGBClassifier()

xgb.fit(X_train, Y_train)

print('Ran Successfully')
#Predict usng our model now

predictions1 = xgb.predict(X_test)
#Evaluate Predictions

accuracy = accuracy_score(Y_test, predictions1)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# split data into train and test sets 

seed = 7

test_size = 0.33 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBClassifier()

eval_set = [(X_test, y_test)]



#Set eval_metrics as logloss, early_stopping_round as 5 

model.fit(X_train, y_train, early_stopping_rounds=5, eval_metric="logloss", eval_set=eval_set, verbose=True) 

# make predictions for test data 

y_predictions = model.predict(X_test)  

# evaluate predictions

accuracy = accuracy_score(y_test, y_predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# fit model on training data 

model = XGBClassifier() 

eval_set = [(X_test, y_test)] 

#Set eval_metrics as logloss, early_stopping_round as 5 

model.fit(X_train, y_train, early_stopping_rounds=5, eval_metric="error", eval_set=eval_set, verbose=True) 

# make predictions for test data

y_predictions = model.predict(X_test) 

# evaluate predictions 

accuracy = accuracy_score(y_test, y_predictions) 

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# plot feature importance using built-in function 

# fit model on training data

model = XGBClassifier() 

model.fit(X_train, y_train) 

# plot feature importance 

plot_importance(model) 

pyplot.show()
# Tune learning_rate 

from sklearn.model_selection import GridSearchCV 

from sklearn.model_selection import KFold, StratifiedKFold
#Split Dataset

X = dataset[:,0:8] 

Y = dataset[:,8] 

# grid search 

model = XGBClassifier() 

learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3] 

param_grid = dict(learning_rate=learning_rate) 

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7) 

grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold) 

grid_result = grid_search.fit(X, Y) 
# summarize results 

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 

means = grid_result.cv_results_['mean_test_score'] 

stds = grid_result.cv_results_['std_test_score'] 

params = grid_result.cv_results_['params'] 

for mean, stdev, param in zip(means, stds, params): 

    print("%f (%f) with: %r" % (mean, stdev, param))

print ('Thank you all for stopping by to learn and as you comment')
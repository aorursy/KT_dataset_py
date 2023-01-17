import pandas as pd

import numpy as np

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBRegressor

from sklearn.datasets import make_classification

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt

%matplotlib inline
train_path = '../input/learn-together/train.csv'

test_path = '../input/learn-together/test.csv'

train_df = pd.read_csv(train_path, index_col='Id')

test_df = pd.read_csv(test_path, index_col='Id')

train_df.head()
test_df.head()
train_df.dtypes
# Print columns name

train_df.columns
X = train_df.drop('Cover_Type', axis = 1)

y = train_df['Cover_Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)
rfc = RandomForestClassifier(n_estimators=100)



rfc.fit(X_train, y_train)



rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print(mean_absolute_error(rfc_pred, y_test))
no_estimators = [20,30,40,50,60,80,100,120,140,160,180,200,210,240,260,280,300,350,400, 420]
def getmae(X,y,K,v):

    rf = RandomForestClassifier(n_estimators=i)

    rf.fit(X, y)

    rf_pred = rf.predict(K)

    mae = mean_absolute_error(rf_pred, v)

    print('With no of estimators =' + str(i) + ',' + ' mae =' + str(mae))

    

    

    

    
for i in no_estimators:

    getmae(X_train, y_train, X_test, y_test)
# It appears using 350 estimators offers a slightly better mean absolute error, but not highly significant.



rfc = RandomForestClassifier(n_estimators=100)



rfc.fit(X, y)

from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print(mean_absolute_error(svm_pred, y_test))
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train, y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(mean_absolute_error(grid_predictions, y_test))
svm_model2 = SVC()
svm_model2.fit(X, y)
param_grid2 = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
grid2 = GridSearchCV(SVC(),param_grid2,refit=True,verbose=3)
grid2.fit(X, y)
grid2.best_params_
grid2.best_estimator_
preds2 = grid2.predict(test_df)
# Get predictions

preds = rfc.predict(test_df)

test_ids = test_df.index



output = pd.DataFrame({'Id': test_ids,

                       'Cover_Type': preds2})

output.to_csv('submission.csv', index=False)



output.head()

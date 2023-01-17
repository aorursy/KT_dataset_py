# Ref : https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

import sklearn.metrics as mtrcs

import os

import matplotlib as plt

from sklearn.model_selection import GridSearchCV



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/HeartDisease.csv')

#Lets see the structure of the data



print(data.info())



print(data.head())
target = data['num'].as_matrix().reshape([457,1])


dummy_cat =pd.get_dummies(data['Place'], drop_first = True) #dropping the one column of the dummy varables as it is redundant



#selecting the varables

#well, in this case all



predictor_df =data[['Age','Sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']]



predictor_df = pd.concat([predictor_df,dummy_cat],axis = 1)



print(predictor_df.head())



predictors = predictor_df.as_matrix()
seed = 1

test_size = 0.33

X_train, X_test, y_train, y_test = train_test_split(predictors,target, test_size=test_size, random_state=seed)
model = XGBClassifier() # declaring the XGBoost classifier
grid = GridSearchCV(estimator=model,scoring = 'accuracy',param_grid={'learning_rate': np.array([0.1,0.2,0.3,0.6,0.7]),'max_depth':np.array([2,3,4,5,6])})

grid.fit(X_train, y_train)

print(grid)

# summarize the results of the grid search

print(grid.best_score_)
print(grid.best_estimator_)
y_pred = grid.best_estimator_.predict(X_test)

predictions = [round(value) for value in y_pred]
accuracy = mtrcs.accuracy_score(y_test, predictions)

recall = mtrcs.recall_score(y_test,predictions)

precision = mtrcs.precision_score(y_test,predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

print("Recall: %.2f%%" % (recall * 100.0))

print("Precision: %.2f%%" % (precision * 100.0))
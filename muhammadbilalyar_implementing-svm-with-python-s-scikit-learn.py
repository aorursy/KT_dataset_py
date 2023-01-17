import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

#import SVC model and initiate it with empty variable

from sklearn.svm import SVC

#Import data set in a variable 'cancer'

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()



#Load input features as DataFrame

df_features = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])



#Add output variable 'target' into Data Frame 

df_target = pd.DataFrame(cancer['target'], columns = ['Cancer'])
#print all columns (features and lable)

df_features.columns
#print dataframe feature

df_features.head()
#print dataframe Lables

df_target.head()
#import 'train_test_split' function to split the data set.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_features, np.ravel(df_target), test_size=0.30, random_state=101)
model = SVC()



#Train the model using fit method

model.fit(X_train, y_train)

predictions = model.predict(X_test)



print(classification_report(y_test, predictions))

#Gridsearch

param_grid = {'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001], 'kernel':['rbf']}



grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)



grid.fit(X_train, y_train)

print('The best parameters are ', grid.best_params_)
grid_predictions = grid.predict(X_test)



print(classification_report(y_test, grid_predictions))

l_param_grid = {'C':[100], 'gamma':[1], 'kernel':['linear']}



l_grid = GridSearchCV(SVC(), l_param_grid, refit=True, verbose=3)



l_grid.fit(X_train, y_train)



l_grid_predictions = l_grid.predict(X_test)



print(classification_report(y_test, l_grid_predictions))
model = SVC(kernel='sigmoid')



#Train the model using fit method

model.fit(X_train, y_train)



predictions = model.predict(X_test)



print(classification_report(y_test, predictions))
#Gridsearch

s_param_grid = {'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001], 'kernel':['sigmoid']}



s_grid = GridSearchCV(SVC(), s_param_grid, refit=True, verbose=3)



s_grid.fit(X_train, y_train)

print('The best parameters are ', s_grid.best_params_)
s_grid_predictions = s_grid.predict(X_test)



print(classification_report(y_test, s_grid_predictions))

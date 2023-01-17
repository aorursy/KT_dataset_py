#imports:
import pandas as pd
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
filename = "../input/ml2020-zsbnn-uibk/train_set.csv"
data_train = pd.read_csv(filename)
filename = "../input/ml2020-zsbnn-uibk/test_set.csv"
data_test = pd.read_csv(filename)
fea_col = data_train.columns[2:]

data_Y = data_train['target']
data_X = data_train[fea_col]
data_train.describe().T
#First, we need to replace the missing values ( = -1) with some data, in order to get more usable data:
# The missing values are replaces with the mean of their row
simpleImputerMean = SimpleImputer(missing_values=-1, strategy = 'mean')
clean_Data_X = simpleImputerMean.fit_transform(data_X)
# Then, we have to center the Data:
standardScaler = StandardScaler()
clean_Data_X = standardScaler.fit_transform(clean_Data_X)
clean_Data_X = pd.DataFrame(data=clean_Data_X, columns = fea_col)
clean_Data_X.describe().T
# In order to get better numbers inside of the featueres,, we also apply a minmaxscaler:
minMaxScaler = MinMaxScaler()
clean_Data_X = minMaxScaler.fit_transform(clean_Data_X)
clean_Data_X = pd.DataFrame(data = clean_Data_X, columns = fea_col)
clean_Data_X.describe().T
x_train, x_val, y_train, y_val = train_test_split(clean_Data_X, data_Y, test_size = 0.3, shuffle = True)
pipe = Pipeline([
        ('fill', SimpleImputer(missing_values=-1, strategy='mean')),
        ('std_scaler', StandardScaler()),
        ('min_max', MinMaxScaler()),
        ('model', DecisionTreeClassifier())
])

param_grid = dict(
                  model__max_depth=[5, 10, 15, 20],
                  model__min_samples_split=[10,20,30,40,50],
                  model__min_samples_leaf=[5,10,15,20,25,30],
                  model__class_weight=[{0:1, 1:5}, {0:1, 1:10}, {0:1, 1:15},  {0:1, 1:20}],
                )

grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=1, verbose=2, scoring='f1_macro')
grid.fit(x_train, y_train)
print(grid.score(x_val, y_val))
print(grid.best_params_)
param_grid2 = dict(
                  model__max_depth=[4,5,6],
                  model__min_samples_split=[18,20,22],
                  model__min_samples_leaf=[4,5,6],
                  model__class_weight=[{0:1, 1:14}, {0:1, 1:15},  {0:1, 1:16}],
                )

grid = GridSearchCV(pipe, param_grid=param_grid2, cv=3, n_jobs=1, verbose=2, scoring='f1_macro')
grid.fit(x_train, y_train)
print(grid.score(x_val, y_val))
print(grid.best_params_)
kfold = KFold(n_splits=5, shuffle=True)
results = cross_val_score(pipe, data_X, data_Y, cv=kfold, scoring='f1_macro')
print(results.mean())
filter_fea = data_train.columns[2:39]

data_X = data_train[filter_fea]
data_Y = data_train["target"]
dparams = {'class_weight': {0: 1, 1: 15}, 'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 20}
dpipe = Pipeline([
        ('fill', SimpleImputer(missing_values=-1, strategy='mean')),
        ('std_scaler', StandardScaler()),
        ('min_max', MinMaxScaler()),
        ('model', DecisionTreeClassifier(**dparams))
])
dpipe.fit(data_X, data_Y)
kfold = KFold(n_splits=5, shuffle=True)
dresults = cross_val_score(dpipe, data_X, data_Y, cv=kfold, scoring='f1_macro')
dresults.mean()
data_test_X = data_test[filter_fea]
y_target = dpipe.predict(data_test_X)
print(sum(y_target==0))
print(sum(y_target==1))
data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True) 
data_out.to_csv('submission.csv',index=False)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import sys, os, scipy, sklearn

import sklearn.metrics, sklearn.preprocessing, sklearn.model_selection, sklearn.tree, sklearn.linear_model, sklearn.cluster, sklearn.ensemble
mpl.rcParams['font.size'] = 14

pd.options.display.max_columns = 1000
data_folder = './'

data_files = os.listdir(data_folder)

display('Course files:',

        data_files)

for file_name in data_files:

    if '.csv' in file_name:

        globals()[file_name.replace('.csv','')] = pd.read_csv(data_folder+file_name, 

                                                              ).reset_index(drop=True)

        print(file_name)

        display(globals()[file_name.replace('.csv','')].head(), globals()[file_name.replace('.csv','')].shape)
import os

print(os.listdir("../input"))
indian_liver_patient = pd.read_csv('../input/indian_liver_patient.csv')

df = indian_liver_patient.rename(columns={'Dataset':'Liver_disease'})

df = df.dropna()
X = df[['Age', 'Total_Bilirubin', 

        'Direct_Bilirubin',

        'Alkaline_Phosphotase',

        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',

       'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio', 'Gender']]

y = df['Liver_disease']-1
LabelEncoder = sklearn.preprocessing.LabelEncoder()

X['Is_male'] = LabelEncoder.fit_transform(X['Gender'])

X = X.drop(columns='Gender')
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)

print(X_train.shape,y_train.shape)
# Define params_dt

params_dt = {'max_depth':[2,3,4], 'min_samples_leaf':[0.12,0.14,0.16,0.18]}
# Import GridSearchCV

from sklearn.model_selection import GridSearchCV



dt = sklearn.tree.DecisionTreeClassifier()



# Instantiate grid_dt

grid_dt = GridSearchCV(estimator=dt,

                       param_grid=params_dt,

                       scoring='roc_auc',

                       cv=5,

                       n_jobs=-1)
grid_dt.fit(X_train, y_train)
# Import roc_auc_score from sklearn.metrics

from sklearn.metrics import roc_auc_score



# Extract the best estimator

best_model = grid_dt.best_estimator_



# Predict the test set probabilities of the positive class

y_pred_proba = best_model.predict_proba(X_test )[:,1]



# Compute test_roc_auc

test_roc_auc = roc_auc_score(y_test, y_pred_proba)



# Print test_roc_auc

print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))
# Define the dictionary 'params_rf'

params_rf = {'n_estimators':[100, 350, 500],

             'max_features':['log2','auto','sqrt'],

             'min_samples_leaf':[2,10,30]}
# Import GridSearchCV

from sklearn.model_selection import GridSearchCV



rf = sklearn.ensemble.RandomForestRegressor()



# Instantiate grid_rf

grid_rf = GridSearchCV(estimator=rf,

                       param_grid=params_rf,

                       scoring='neg_mean_squared_error',

                       cv=3,

                       verbose=1,

                       n_jobs=-1)
grid_rf.fit(X_train, y_train)
# Import mean_squared_error from sklearn.metrics as MSE 

from sklearn.metrics import mean_squared_error as MSE



# Extract the best estimator

best_model = grid_rf.best_estimator_



# Predict test set labels

y_pred = best_model.predict(X_test)



# Compute rmse_test

rmse_test = MSE(y_test, y_pred)**0.5



# Print rmse_test

print('Test RMSE of best model: {:.3f}'.format(rmse_test)) 
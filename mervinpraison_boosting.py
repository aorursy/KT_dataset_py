import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import sys, os, scipy, sklearn

import sklearn.metrics, sklearn.preprocessing, sklearn.model_selection, sklearn.tree, sklearn.linear_model, sklearn.cluster
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
indian_liver_patient = pd.read_csv('../input/indianliver/indian_liver_patient.csv')

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
# Import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier



# Import AdaBoostClassifier

from sklearn.ensemble import AdaBoostClassifier



# Instantiate dt

dt = DecisionTreeClassifier(max_depth=2, random_state=1)



# Instantiate ada

ada = AdaBoostClassifier(base_estimator=dt, 

n_estimators=180, random_state=1)
# Fit ada to the training set

ada.fit(X_train, y_train)



# Compute the probabilities of obtaining the positive class

y_pred_proba = ada.predict_proba(X_test)[:,1]
# Import roc_auc_score

from sklearn.metrics import roc_auc_score



# Evaluate test-set roc_auc_score

ada_roc_auc = roc_auc_score(y_test, y_pred_proba)



# Print roc_auc_score

print('ROC AUC score: {:.2f}'.format(ada_roc_auc))
# Import GradientBoostingRegressor

from sklearn.ensemble import GradientBoostingRegressor



# Instantiate gb

gb = GradientBoostingRegressor(max_depth=4, 

            n_estimators=200,

            random_state=2)
# Fit gb to the training set

gb.fit(X_train,y_train)



# Predict test set labels

y_pred = gb.predict(X_test)
# Import mean_squared_error as MSE

from sklearn.metrics import mean_squared_error as MSE



# Compute MSE

mse_test = MSE(y_test, y_pred)



# Compute RMSE

rmse_test = mse_test**0.5



# Print RMSE

print('Test set RMSE of gb: {:.3f}'.format(rmse_test))
# Import GradientBoostingRegressor

from sklearn.ensemble import GradientBoostingRegressor



# Instantiate sgbr

sgbr = GradientBoostingRegressor(

            max_depth=4, 

            subsample=0.9,

            max_features=0.75,

            n_estimators=200,                                

            random_state=2)
# Fit sgbr to the training set

sgbr.fit(X_train,y_train)



# Predict test set labels

y_pred = sgbr.predict(X_test)
# Import mean_squared_error as MSE

from sklearn.metrics import mean_squared_error as MSE



# Compute test set MSE

mse_test = MSE(y_test,y_pred)



# Compute test set RMSE

rmse_test = mse_test**0.5



# Print rmse_test

print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))
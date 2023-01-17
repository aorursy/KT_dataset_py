# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Any results you write to the current directory are saved as output.
#data setup
from sklearn.ensemble import RandomForestRegressor

#name data
dproject= pd.read_csv('../input/1 - Project.csv')
dcost= pd.read_csv('../input/2 - Cost.csv')
dcostp= pd.read_csv('../input/2B - Cost and Project.csv')
dplan= pd.read_csv('../input/3 - Planning.csv')
dplanp= pd.read_csv('../input/3B - Planning and Project.csv')
dmile= pd.read_csv('../input/4 - Milestone.csv')
dmilep= pd.read_csv('../input/4B - Milestone and Project.csv')

#pull data into target (y) and predictors (X)
dplanp['Success?']= dplanp['Success?'].map({'Yes':1, 'No':0})
y= dplanp['Success?']
predictor_cols= ['Cumul_Planned','Period_Planned','Cumul_Earned','Period_Earned']
#Create training predictors data
X= dplanp[predictor_cols]

#split that training data into training and validation

#------------------------------
#code to measure performance score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)



#codeforimputation
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))

#code for imputation extension
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))



dtest= pd.read_csv('../input/3B - Planning and Project.csv')
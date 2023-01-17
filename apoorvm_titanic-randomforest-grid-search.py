# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

import random

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
def model():

    

    # Create the parameter grid based on the results of random search 

#     param_grid = {

#         'bootstrap': [True, False],

#         'max_depth': [80, 90, 100, 110],

#         'max_features': [2, 3],

#         'min_samples_leaf': [3, 4, 5],

#         'min_samples_split': [8, 10, 12],

#         'n_estimators': [100, 200, 300, 1000]

#     }



    param_grid = {'bootstrap': False, 'max_depth': 80, 'max_features': 3,

                  'min_samples_leaf': 5, 'min_samples_split': 8, 'n_estimators': 200}



    # Create a base model

    rf = RandomForestRegressor(random_state = 42,bootstrap = False, max_depth= 80, max_features = 3,

                  min_samples_leaf = 5, min_samples_split = 8, n_estimators = 200)



    # Instantiate the grid search model

#     grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

#                           cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)

    

    return rf
def preprocess(X, X_test):

    

    #Checking for number of missing values

    print('Missing values in Train and Test')

    (X.isnull().sum())

    (X_test.isnull().sum())

    

    #Categorical Columns

    categorical_cols = [cname for cname in X.columns if

                    X[cname].nunique() < 10 and 

                    X[cname].dtype == "object"]

    

    #Numerical Columns

    numerical_cols = [cname for cname in X.columns if 

                X[cname].dtype in ['int64', 'float64']]

    

    print(categorical_cols)

    print(numerical_cols)

    

    #Preprocess the numerical data

    numerical_transformer = SimpleImputer(strategy='constant')

    

    #Preprocess the categorical data

    categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

    ])

    

    # Bundle preprocessing for numerical and categorical data

    preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])

    

    new_cols = categorical_cols + numerical_cols

    

    # Keep selected columns only

    X_train = X[new_cols].copy()

    X_test = X_test[new_cols].copy()

    

#     print(X_train.isnull().sum())

#     print(X_test.isnull().sum())



    

    

    return preprocessor, X_train, y, X_test
def train_and_predict(X, y, X_test):

    preprocessor, X_train, y, X_test = preprocess(X, X_test)

    

    rf_classifier = model()

    

    # Bundle preprocessing and modeling code in a pipeline

    rf_classifier_model = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', rf_classifier)

                         ])

    

    #Fit the model

#     grid_rf_classifier_model.fit(X_train, y)

    rf_classifier_model.fit(X_train, y)

#     print(grid_rf_classifier.best_params_) #for finding best parameters

    

    

    #get predictions

    y_pred = rf_classifier_model.predict(X_test)

    

    y_pred = np.around(y_pred)

    y_pred = y_pred.astype(int)



    return y_pred
def submission_file(y_pred):

    # Save test predictions to file

    output = pd.DataFrame({'PassengerId': test_data.PassengerId,

                           'Survived': y_pred})

    output.to_csv('submission_grid_search.csv', index=False)

#     print(y_pred.astype() np.around(y_pred[:5]))

    return 
if __name__ == '__main__':

    seed = 123

    random.seed(seed)

    

    print ('Loading Training Data')

    data = pd.read_csv('/kaggle/input/titanic/train.csv')

    

    cols = [col for col in data.columns if col not in ['Survived','PassengerId']]

    X = data[cols]

    y = data['Survived']

    

    #Now Laoding the testing data

    print ('Loading Testing Data')

    test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

    X_test = test_data.iloc[:,1:11]

    

    #Train model now

    submission_file(train_and_predict(X, y, X_test))

    

    
# (X.isnull().sum())

    (X_test.isnull().sum())
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

    

    #     param_grid = {'bootstrap': False, 'max_depth': 80, 'max_features': 3,

#                   'min_samples_leaf': 5, 'min_samples_split': 8, 'n_estimators': 200}



    # Create a base model

    rf = RandomForestRegressor(random_state = 42,bootstrap = False, max_depth= 80, max_features = 2,

                  min_samples_leaf = 5, min_samples_split = 8, n_estimators = 100)



    # Instantiate the grid search model

#     grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

#                           cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)

    

    return rf
def preprocess(X, X_test):

    

    #Checking for number of missing values

    print('Missing values in Train')

    print(X.isnull().sum())

    print('Missing values in Test')

    print(X_test.isnull().sum())

    

    #Categorical Columns

    categorical_cols = [cname for cname in X.columns if 

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

    

    

    pipe = Pipeline(steps=[('preprocessor', preprocessor)])

    new_cols = categorical_cols + numerical_cols

    

    print("new cols: ", new_cols)

    

    # Keep selected columns only

    X_train = X[new_cols].copy()

#     categorical_transformer.fit(X_train)

#     categorical_transformer.transform(X_train)

    X_test = X_test[new_cols].copy()

    

#     print(X_train.isnull().sum())

#     print(X_test.isnull().sum())



    

    

    return preprocessor, X_train, X_test
def train_and_predict(X, y, X_test):

    preprocessor, X_train, X_test = preprocess(X, X_test)

    

    rf_classifier = model()

    

    # Bundle preprocessing and modeling code in a pipeline

    rf_classifier_model = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', rf_classifier)

                         ])

    

    #Fit the model

#     grid_rf_classifier_model.fit(X_train, y)

    rf_classifier_model.fit(X, y)

#     print(grid_rf_classifier.best_params_) #for finding best parameters

    

    

    #get predictions

    y_pred = rf_classifier_model.predict(X_test)

    

    y_pred = np.around(y_pred)

    y_pred = y_pred.astype(int)



    return y_pred
if __name__ == '__main__':

    seed = 123

    random.seed(seed)

    

    print ('Loading Training Data')

    covid_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv",

                          parse_dates=['Date'])

    

#     # extracting date from timestamp

#     covid_train['year'] = pd.DatetimeIndex(covid_train.Date).year

#     covid_train['month'] = pd.DatetimeIndex(covid_train.Date).month

#     covid_train['day'] = pd.DatetimeIndex(covid_train.Date).day

    covid_train = covid_train.drop(['Province_State'], axis=1)

    covid_train = covid_train.drop(['Country_Region'], axis=1)

    

# #     covid_train.fillna('0', inplace=True)

    covid_train['Date'] = (pd.to_datetime(covid_train['Date'], unit='s').astype(int)/10**9).astype(int)

    cols = [col for col in covid_train.columns if col not in ['Id','ConfirmedCases','Fatalities']]

    X = covid_train[cols]

    y1 = covid_train['ConfirmedCases']

    y2 = covid_train['Fatalities']

    

    #Now Laoding the testing data

    print ('Loading Testing Data')

    covid_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv",

                         parse_dates=['Date'])

#     # extracting date from timestamp

#     covid_test['year'] = pd.DatetimeIndex(covid_test.Date).year

#     covid_test['month'] = pd.DatetimeIndex(covid_test.Date).month

#     covid_test['day'] = pd.DatetimeIndex(covid_test.Date).day

    covid_test = covid_test.drop(['Province_State'], axis=1)

    covid_test = covid_test.drop(['Country_Region'], axis=1)

    

#     covid_train.fillna('0', inplace=True)

    covid_test['Date'] = (pd.to_datetime(covid_test['Date'], unit='s').astype(int)/10**9).astype(int)

    X_test = covid_test.iloc[:,1:]

    

# #     preprocessor, X_train, X_test = preprocess(X, X_test)

    

    

    confirmer_y_pred = train_and_predict(X, y1, X_test)

    

    fatality_y_pred = train_and_predict(X, y2, X_test)

    

    pred=pd.DataFrame()

    pred['ForecastId']=covid_test['ForecastId']

    pred['ConfirmedCases']=confirmer_y_pred

    pred['Fatalities']=fatality_y_pred

    pred.to_csv('submission.csv',index=False)
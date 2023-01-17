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
import datetime
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
def train_and_predict(X, y, X_test):

#     preprocessor, X_train, y, X_test = preprocess(X, X_test)

    

    rf_classifier = model()

    

    # Bundle preprocessing and modeling code in a pipeline

    rf_classifier_model = Pipeline(steps=[

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

    covid_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv",

                          parse_dates=['Date'])

    

#     # extracting date from timestamp

#     covid_train['year'] = pd.DatetimeIndex(covid_train.Date).year

#     covid_train['month'] = pd.DatetimeIndex(covid_train.Date).month

#     covid_train['day'] = pd.DatetimeIndex(covid_train.Date).day

    covid_train = covid_train.drop(['Province/State'], axis=1)

    covid_train = covid_train.drop(['Country/Region'], axis=1)

    

# #     covid_train.fillna('0', inplace=True)

    covid_train['Date'] = (pd.to_datetime(covid_train['Date'], unit='s').astype(int)/10**15).astype(int)

#     covid_train['Date'] = str(covid_train['Date'])

    cols = [col for col in covid_train.columns if col not in ['Id','ConfirmedCases','Fatalities']]

    X = covid_train[cols]

    y1 = covid_train['ConfirmedCases']

    y2 = covid_train['Fatalities']

    

    #Now Laoding the testing data

    print ('Loading Testing Data')

    covid_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv",

                         parse_dates=['Date'])

#     # extracting date from timestamp

#     covid_test['year'] = pd.DatetimeIndex(covid_test.Date).year

#     covid_test['month'] = pd.DatetimeIndex(covid_test.Date).month

#     covid_test['day'] = pd.DatetimeIndex(covid_test.Date).day

    covid_test = covid_test.drop(['Province/State'], axis=1)

    covid_test = covid_test.drop(['Country/Region'], axis=1)

    

#     covid_train.fillna('0', inplace=True)

    covid_test['Date'] = (pd.to_datetime(covid_test['Date'], unit='s').astype(int)/10**15).astype(int)

#     covid_test['Date'] = str(covid_test['Date'])

    X_test = covid_test.iloc[:,1:]

    

# #     preprocessor, X_train, X_test = preprocess(X, X_test)

    

    

    confirmer_y_pred = train_and_predict(X, y1, X_test)

    

    fatality_y_pred = train_and_predict(X, y2, X_test)

    

    pred=pd.DataFrame()

    pred['ForecastId']=covid_test['ForecastId']

    pred['ConfirmedCases']=confirmer_y_pred

    pred['Fatalities']=fatality_y_pred

    pred.to_csv('submission.csv',index=False)



    
covid_train['Date'] = (pd.to_datetime(covid_train['Date'], unit='s').astype(int)/10**9).astype(int)
covid_train.head()
covid_train.head()
fatality_y_pred
str(covid_train['Date'][0])
for i in range(len(covid_train['Date'])):

    covid_train['Date'][i] = covid_train['Date'][i].strftime("%d %B, %Y")
covid_train['Date'] = str(covid_train['Date'])
covid_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv",

                          parse_dates=['Date'])
covid_train['Date'].astype(int) / 10**15
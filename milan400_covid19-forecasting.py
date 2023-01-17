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



from xgboost import XGBRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.impute import SimpleImputer



from shapely.geometry import Point,Polygon

import requests 
train_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

submission_csv = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
convert_dict = {'Province_State': str,'Country_Region':str,'ConfirmedCases':int,'Fatalities':int}

convert_dict_test = {'Province_State': str,'Country_Region':str}

train_data = train_data.astype(convert_dict)

test_data = test_data.astype(convert_dict_test)
train_data['Date'] = pd.to_datetime(train_data['Date'], infer_datetime_format=True)

test_data['Date'] = pd.to_datetime(test_data['Date'], infer_datetime_format=True)
train_data.loc[:, 'Date'] = train_data.Date.dt.strftime('%m%d')

train_data.loc[:, 'Date'] = train_data['Date'].astype(int)



test_data.loc[:, 'Date'] = test_data.Date.dt.strftime('%m%d')

test_data.loc[:, 'Date'] = test_data['Date'].astype(int)
train_data['Country_Region'] = np.where(train_data['Province_State'] == 'nan',train_data['Country_Region'],train_data['Province_State']+' '+train_data['Country_Region'])

test_data['Country_Region'] = np.where(test_data['Province_State'] == 'nan',test_data['Country_Region'],test_data['Province_State']+' '+test_data['Country_Region'])



#train_data['Province_State'] = np.where(train_data['Province_State'] == 'nan',train_data['Country_Region'],train_data['Province_State']+train_data['Country_Region'])

#test_data['Province_State'] = np.where(test_data['Province_State'] == 'nan',test_data['Country_Region'],test_data['Province_State']+test_data['Country_Region'])





train_data = train_data.drop(columns=['Province_State'])

test_data = test_data.drop(columns=['Province_State'])
test_data.head(2)
#get list of categorical variables

s = (train_data.dtypes == 'object')

object_cols = list(s[s].index)
from sklearn.preprocessing import LabelEncoder
label_encoder1 = LabelEncoder()

label_encoder2 = LabelEncoder()



#train_data['Province_State'] = label_encoder1.fit_transform(train_data['Province_State'])

#test_data['Province_State'] = label_encoder1.transform(test_data['Province_State'])



train_data['Country_Region'] = label_encoder2.fit_transform(train_data['Country_Region'])

test_data['Country_Region'] = label_encoder2.transform(test_data['Country_Region'])



    
train_data.head(2)
test_data.head(2)
Test_id = test_data.ForecastId
train_data.drop(['Id'], axis=1, inplace=True)

test_data.drop('ForecastId', axis=1, inplace=True)
missing_val_count_by_column = (train_data.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column>0])
from xgboost import XGBRegressor
train_data.head(1)
X_train = train_data[['Country_Region','Date']]

y_train = train_data[['ConfirmedCases', 'Fatalities']]
x_train = X_train.iloc[:,:].values

x_test = test_data.iloc[:,:].values
#from sklearn.metrics import mean_squared_error
#X_train,X_test,Y_train,Y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)
error_list = []

def return_error(estimator, x_train,x_test,y_train):

    model = MultiOutputRegressor(XGBRegressor(n_estimators=estimator, random_state=0, max_depth=20))

    model.fit(x_train, y_train)



    predict = MultiOutputRegressor(model.predict(x_test))

    

    #error = mean_squared_error( y_test.values, predict.estimator)

    #error_list.append(error)

    

    return predict
#from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
#num_estimators = [1000,1100,1200,1250,1300]

#learn_rates = [0.02,0.05,0.06,0.07]



#param_grid = {'n_estimators':num_estimators,

 #             'learning_rate':learn_rates

#            }
#random_search = GridSearchCV(XGBRegressor(loss='huber'), param_grid,cv=3,return_train_score=True, n_jobs=1)
#random_search.fit(x_train, y_train.Fatalities)
#random_search.best_params_
#estimator_list = [1200,1250,1300,1350]

#for value in estimator_list:

#    error_ = return_error(value, X_train,X_test,Y_train,Y_test)
predict = return_error(1500,x_train,x_test,y_train)
df_sub = pd.DataFrame()

df_sub['ForecastId'] = Test_id

df_sub['ConfirmedCases'] = (predict.estimator[:,0]).astype(int)

df_sub['Fatalities'] = (predict.estimator[:,1]).astype(int)



df_sub.to_csv('submission.csv', index=False)
df_sub
!pip install catboost
import numpy as np

import pandas as pd

from pandas_profiling import ProfileReport

#import libraries for visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

pd.options.display.max_columns = 100

# libraries for machine learning

from sklearn.model_selection import train_test_split 

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
amsterdam_airbnb = pd.read_csv('../input/dphi-amsterdam-airbnb-data/airbnb_listing_train.csv')
amsterdam_airbnb.columns
amsterdam_airbnb.info()
amsterdam_airbnb.head()
amsterdam_airbnb.describe()
amsterdam_airbnb.isnull().sum()
#dropping name and host_name as we have id and host_id

#dropping neighbourhood_group as entire column has missing value

amsterdam_airbnb.drop(['name','host_name','neighbourhood_group'], axis=1, inplace =True)
#dropping last_review and reviews_per_month as we have number_of_reviews which is more relevant compare to previous two columns

amsterdam_airbnb.drop(['last_review','reviews_per_month'], axis=1, inplace =True)
#convert to category dtype

amsterdam_airbnb['neighbourhood'] = amsterdam_airbnb['neighbourhood'].astype('category')
#use .cat.codes to create new colums with encoded value

amsterdam_airbnb['neighbourhood'] = amsterdam_airbnb['neighbourhood'].cat.codes
amsterdam_airbnb['neighbourhood'].value_counts()
amsterdam_airbnb['room_type'].value_counts()
#convert categorical variable into dummy/indicator variables for ML

amsterdam_airbnb = pd.get_dummies(amsterdam_airbnb, columns=['room_type'])
#Convert longitude and latitude into cartesian coordinates. Assume the earth as sphere not ellipsoid

R = 6371000 #Approximate mean radius of earth (in m)

 #lons and lats must be in radians

lon,lat = map(np.radians, [amsterdam_airbnb['longitude'], amsterdam_airbnb['latitude']])



# 'Single-point' Haversine formula

a = np.sin(lat/2)**2 + np.cos(lat) * np.sin(lon/2)**2

distance = 2 * R * np.arcsin(np.sqrt(a))



#creating new column distance and dropping longitude and latitude column

amsterdam_airbnb['distance'] = distance

amsterdam_airbnb.drop(['latitude', 'longitude'], axis = 1, inplace = True)
amsterdam_airbnb.columns
amsterdam_airbnb.dtypes
amsterdam_airbnb.info()
amsterdam_airbnb.groupby('price').mean()
X = amsterdam_airbnb.drop('price', axis = 1)

y = amsterdam_airbnb['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
from catboost import CatBoostRegressor

from sklearn.model_selection import GridSearchCV
model = CatBoostRegressor(random_seed = 42 )

parameters = {'depth'         : [6,8,10,21],

                  'learning_rate' : [0.01, 0.05, 0.1],

                  'iterations'    : [30, 50, 100]

              }

grid = GridSearchCV(estimator=model, param_grid = parameters, cv = 2, n_jobs=-1)

grid.fit(X_train, y_train)

pred = grid.predict(X_test)
grid.score(X_test, y_test)
r2_score(y_test, pred)
#evaluating the performance of the model

#MAE

print(mean_absolute_error(y_test, pred), end='\n')

#MSE

print(mean_squared_error(y_test, pred), end='\n')

#RMSE

print(np.sqrt(mean_squared_error(y_test, pred)))
test_new = pd.read_csv('../input/dphi-amsterdam-airbnb-data/airbnb_listing_validate.csv')
test_new.columns
test_new.info()
test_new.drop(['name','host_name','neighbourhood_group', 'last_review','reviews_per_month'], axis=1, inplace =True)
#convert to category dtype

test_new['neighbourhood'] = test_new['neighbourhood'].astype('category')

test_new.dtypes
#use .cat.codes to create new colums with encoded value

test_new['neighbourhood'] = test_new['neighbourhood'].cat.codes
#convert categorical variable into dummy/indicator variables for ML

test_new = pd.get_dummies(test_new, columns=['room_type'])
#Convert longitude and latitude into cartesian coordinates. Assume the earth as sphere not ellipsoid

R = 6371000 #Approximate mean radius of earth (in m)

 #lons and lats must be in radians

lon,lat = map(np.radians, [test_new['longitude'], test_new['latitude']])



# 'Single-point' Haversine formula

a = np.sin(lat/2)**2 + np.cos(lat) * np.sin(lon/2)**2

distance = 2 * R * np.arcsin(np.sqrt(a))



#creating new column distance and dropping longitude and latitude column

test_new['distance'] = distance

test_new.drop(['latitude', 'longitude'], axis = 1, inplace = True)
test_new.columns
test_new.dtypes
test_new.info()
newmodel = CatBoostRegressor(random_seed = 42 )

newparameters = {'depth'         : [6,8,10,21],

                  'learning_rate' : [0.01, 0.05, 0.1],

                  'iterations'    : [30, 50, 100]

              }

newgrid = GridSearchCV(estimator=newmodel, param_grid = newparameters, cv = 2, n_jobs=-1)

newgrid.fit(X, y)
testnew_pred = newgrid.predict(test_new)
res = pd.DataFrame(testnew_pred) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data

res.index = test_new.index # its important for comparison # its important for comparison. Here "test_new" is your new test dataset

res.index = test_new['id'] # format for Kaggle

res.columns = ['price']

# To download the csv file locally

res.to_csv('/kaggle/working/sample_submission.csv')         
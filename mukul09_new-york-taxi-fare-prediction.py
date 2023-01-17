import numpy as np

import pandas as pd



from math import radians, cos, sin, asin, sqrt



from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

from sklearn.model_selection import ShuffleSplit, KFold

from sklearn.model_selection import GridSearchCV





import warnings

warnings.filterwarnings("ignore")
original_train_data = pd.read_csv('../input/train.csv', nrows=6000000)

train_data = original_train_data.sample(n=100000)

train_data.info()
test_data = pd.read_csv('../input/test.csv')

test_data.info()
## Null values

train_data.isnull().sum()
## Remove rows with null values of dropoff_lat and lan

train_data.dropna(axis=0,inplace=True)

#train_data.shape  (5999961, 8)
## removing outliers

train_data.describe()
# i)

train_data = train_data[train_data['fare_amount']>0]



# ii)

#train_data[train_data['passenger_count']>6].shape   # only 12 rows therefore we can delete them

train_data = train_data[(train_data['passenger_count']<=6)& (train_data['passenger_count']>0)]



# iii)

train_data = train_data[(train_data['pickup_latitude']>-90)| (train_data['pickup_latitude']<=90)]

train_data = train_data[(train_data['dropoff_latitude']>-90)| (train_data['dropoff_latitude']<=90)]



#iv)

train_data = train_data[(train_data['pickup_longitude']>=-180)| (train_data['pickup_longitude']<=180)]

train_data = train_data[(train_data['dropoff_longitude']>=-180)| (train_data['dropoff_longitude']<=180)]





#train_data = train_data[(train_data['pickup_longitude']<-180)| (train_data['pickup_longitude']>180)]

train_data.shape
train_data.info()
train_data.head(5)


def distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):

    """

    Return distance along great radius between pickup and dropoff coordinates.

    """

    #Define earth radius (km)

    R_earth = 6371

    #Convert degrees to radians

    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,

                                                             [pickup_lat, pickup_lon, 

                                                              dropoff_lat, dropoff_lon])

    #Compute distances along lat, lon dimensions

    dlat = dropoff_lat - pickup_lat

    dlon = dropoff_lon - pickup_lon

    

    #Compute haversine distance

    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2

    

    return 2 * R_earth * np.arcsin(np.sqrt(a))

    

def date_time_info(data):

    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'], format="%Y-%m-%d %H:%M:%S UTC")

    

    data['hour'] = data['pickup_datetime'].dt.hour

    data['day']  = data['pickup_datetime'].dt.day

    data['month'] = data['pickup_datetime'].dt.month

    data['weekday'] = data['pickup_datetime'].dt.weekday

    data['year']    = data['pickup_datetime'].dt.year

    

    return data





train_data = date_time_info(train_data)

train_data['distance'] = distance(train_data['pickup_latitude'], 

                                     train_data['pickup_longitude'],

                                     train_data['dropoff_latitude'] ,

                                     train_data['dropoff_longitude'])



train_data.head()
train_data.drop(['key', 'pickup_datetime'],axis =1, inplace = True)

train_data.head()
test_data.head()
test_data = date_time_info(test_data)

test_data['distance'] = distance(test_data['pickup_latitude'], test_data['pickup_longitude'], 

                                   test_data['dropoff_latitude'] , test_data['dropoff_longitude'])



test_key = test_data['key']

x_pred = test_data.drop(columns=['key', 'pickup_datetime'])
y = train_data['fare_amount']

X = train_data.drop(['fare_amount'],axis=1)



cv_split = KFold(n_splits=10,random_state=0)

xgb = XGBRegressor(random_state=0)

base_results = cross_validate(xgb, X,y, cv = cv_split)

xgb.fit(X,y)



print('Best XGB parameters: ', xgb.get_params())

print('Before XGB Training score mean: {:.2f}'.format(base_results['train_score'].mean()*100))

print('Before XGB Training score mean: {:.2f}'.format(base_results['test_score'].mean()*100))

print('#'*20)



# param_grid={

#     'learning_rate': [.01, ],

#     'max_depth'    : [6,8],

#     'n_estimators' : [10],}

# tune_model = GridSearchCV(XGBRegressor(),param_grid=param_grid,

#                                          scoring='neg_mean_squared_error',

#                                          cv=cv_split)

# tune_model.fit(X,y)



print('After XGB Parameters: ', tune_model.best_params_)

print('After XGB Training score mean: {:.2f}'.format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))

print('After XGB Testing score mean: {:.2f}'.format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_].mean()*100))

print('#'*20)
params = {

    # Parameters that we are going to tune.

    'max_depth': [8], #Result of tuning with CV

    'eta':[.03], #Result of tuning with CV

    'subsample': [1], #Result of tuning with CV

    'colsample_bytree': [0.8], #Result of tuning with CV

    # Other parameters

    'objective':['reg:linear'],

    'eval_metric':['rmse'],

    'silent': [1]

}



#submit_xgb = XGBRegressor()

submit_xgb = GridSearchCV(XGBRegressor(), param_grid=params,

                                         scoring = 'neg_mean_squared_error',

                          cv = cv_split)

submit_xgb.fit(X,y)

prediction = submit_xgb.predict(x_pred)
submission = pd.DataFrame({

        "key": test_key,

        "fare_amount": prediction.round(2)

})



submission.to_csv('taxi_fare_submission.csv',index=False)

submission.head()
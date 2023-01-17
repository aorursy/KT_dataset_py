import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge, ElasticNet, LinearRegression,Lasso

from sklearn.model_selection import train_test_split, KFold



import matplotlib.pyplot as plt



from matplotlib.pyplot import figure

%matplotlib inline

from math import sqrt

df = pd.read_csv('../input/uio_clean.csv')

df.head()
df.info()
#dropping the columns that are not useful for prediction

df= df.drop(['id','vendor_id','store_and_fwd_flag'], axis=1)

df.describe()

# removing outliers or erroneous values i.e. trip duration should be between 20 sec and 3hrs, distance should be

# between 100m and 100km, trip duration should be greater than wait time etc.

df=df[(df['trip_duration'].between(30,7200)) & (df['dist_meters'].between(100,100000)) & (df['trip_duration']>df['wait_sec'])]

df=df[df['wait_sec'].between(0,7200)]

df=df[(df['pickup_longitude'].between(-80,-77)) & (df['pickup_latitude'].between(-4,1)) & (df['dropoff_longitude'].between(-80,-77))

      &(df['dropoff_latitude'].between(-4,1))]

df.shape[0]
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')

df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')

df['hour_of_day']=df['pickup_datetime'].dt.hour

df['month'] = df['pickup_datetime'].dt.month

df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

df['day_of_year'] = df['pickup_datetime'].dt.dayofyear

df['week_of_year'] = df['pickup_datetime'].dt.weekofyear

df=df.drop(['pickup_datetime','dropoff_datetime'], axis=1)



                                    

df['trip_duration_log'] = np.round(np.log1p(df['trip_duration']), 5)

df['dist_meters_log'] = np.round(np.log1p(df['dist_meters']), 5)

df['avg_speed'] = df['dist_meters'] / df['trip_duration'] 

# avg speed should be between 3m/s and 30m/s or 108km/hr

df = df[df['avg_speed'].between(3,30)]

df=df.dropna()

df.describe()
#split dependent and independent variable

X = df.drop(['wait_sec'],axis=1)

#Taking natural log of the target variable, this helps the model converge better and gives better results

y = np.log1p(df['wait_sec'])



#Normalization function

scaler = StandardScaler()

#X = scaler.fit_transform(X)



#Test train split

X_train, X_test, y_train, y_test = train_test_split(scaler.fit_transform(X), y, test_size=0.20, random_state=16)
grid_search = GridSearchCV( 

          estimator= ElasticNet(),

        param_grid={

            'alpha':[0.01,0.03,0.001,0.003,0.1],

            'l1_ratio': [0.3,0.35,0.4,0.5]

        },

          scoring="neg_mean_squared_error",  

          cv=KFold(n_splits=3,shuffle=True,random_state=42))   

        

grid_search.fit(X_train, y_train.values.ravel())

CVed_model = grid_search.best_estimator_

print(grid_search.best_params_)
print("The training error is:",grid_search.best_score_)
y_pred = CVed_model.predict(X_test)

mae = np.abs((y_test - y_pred))

print("The test error is: ",mae.mean())
feat_importances = pd.Series(CVed_model.coef_, index=X.columns)

feat_importances.abs().nlargest(10).plot(kind='barh',title='Feature importance')

#taking the inverse log to measure actual error

y_test_act = np.expm1(y_test)

y_pred_act = np.expm1(y_pred)

mae = np.abs((y_test_act - y_pred_act))



print("The actual mean absolute test error is: ",mae.mean())
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
sample=pd.read_csv('../input/sample_submission.csv')

test=pd.read_csv('../input/test.csv',low_memory=False)

train=pd.read_csv('../input/train.csv',low_memory=False)
print(sample.head())

print('\n',test.head())

print('\n',train.head())
print(test.shape)

print(train.shape)
print(test.columns)

print(train.columns)
print(test.isnull().sum())

print(train.isnull().sum())
print(test.info())

print(train.info())

print('\n\n',test.describe(),'\n\n',train.describe())
#Converting Trip duration in Hours

train['trip_dur_hr']=train['trip_duration']/3600

print(train['trip_dur_hr'].describe(),'\n')



#We can see that 75 % of data have value less than or equals to 2.986111e-01 Hrs

#but max value is 9.795228e+02 Hrs(Quite Large) :> Definitly an outlier



#Checking data sufficient for removal

#print(len(train.loc[train['trip_dur_hr']>=4,['trip_dur_hr']])) gives  2077 entries

#print(len(train.loc[train['trip_dur_hr']>=2,['trip_dur_hr']])) gives 2253 entries

#print(len(train.loc[train['trip_dur_hr']>=1,['trip_dur_hr']])) gives 12334 entries

#Hence removing entries with trip_du_hr>=2 seems good



#drop above 2 hrs trip data (removing outliers)

train=train.loc[train['trip_dur_hr']<=2,[str(i) for i in train.columns]]

print(train.shape)

print(min(train['trip_duration']),max(train.trip_duration))

print(min(train.trip_dur_hr),max(train.trip_dur_hr),'\n\n')

print(train['trip_dur_hr'].describe())
# Caculating distance between pickup and dropup location

from math import sin, cos, sqrt, atan2, radians

def distance(lat1,lat2,lon1,lon2):

    R = 6371.0

    lat1 = radians(lat1)

    lon1 = radians(lon1)

    lat2 = radians(lat2)

    lon2 = radians(lon2)

    dlon = lon2 - lon1

    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

train['distance']=[distance(i,j,k,l) for i,j,k,l in zip(train.pickup_latitude,

                                                        train.dropoff_latitude,train.pickup_longitude,

                                                        train.dropoff_longitude)]

test['distance']=[distance(i,j,k,l) for i,j,k,l in zip(test.pickup_latitude,

                                                        test.dropoff_latitude,test.pickup_longitude,

                                                        test.dropoff_longitude)]
#Latitude and Longitude to cartesian cordindates converesion

# Assuming Earth as sphere not ellipsoid 

def cartesian_x(lat,lon):

    lat=radians(lat)

    lon=radians(lon)

    R=6371.0

    x = R * cos(lat) * cos(lon)

    return x

def cartesian_y(lat,lon):

    lat=radians(lat)

    lon=radians(lon)

    R=6371.0

    y = R * cos(lat) * sin(lon)

    return y

# extracting cartesian x,y cordinates form latitude and longitude

train['x1']=[cartesian_x(i,j) for i,j in zip(train['pickup_latitude'],train['pickup_longitude'])]

train['y1']=[cartesian_y(i,j) for i,j in zip(train['pickup_latitude'],train['pickup_longitude'])]

train['x2']=[cartesian_x(i,j) for i,j in zip(train['dropoff_latitude'],train['dropoff_longitude'])]

train['y2']=[cartesian_y(i,j) for i,j in zip(train['dropoff_latitude'],train['dropoff_longitude'])]



test['x1']=[cartesian_x(i,j) for i,j in zip(test['pickup_latitude'],test['pickup_longitude'])]

test['y1']=[cartesian_y(i,j) for i,j in zip(test['pickup_latitude'],test['pickup_longitude'])]

test['x2']=[cartesian_x(i,j) for i,j in zip(test['dropoff_latitude'],test['dropoff_longitude'])]

test['y2']=[cartesian_y(i,j) for i,j in zip(test['dropoff_latitude'],test['dropoff_longitude'])]
#Manhattan Distance

train['Manhattan_dist'] =(train['x1'] - train['x2']).abs() +(train['y1'] - train['y2']).abs()    

test['Manhattan_dist'] =(test['x1'] - test['x2']).abs() + (test['y1'] - test['y2']).abs()   

#Chebyshev Distance

train['Chebyshev_dist']=[max(abs(i-j),abs(k-l)) for i,j,k,l in zip(train['x1'],

                                                                           train['y1'],train['x2'],

                                                                          train['y2'])]

test['Chebyshev_dist']=[max(abs(i-j),abs(k-l)) for i,j,k,l in zip(test['x1'],

                                                                           test['y1'],test['x2'],

                                                                          test['y2'])]

print(train['Chebyshev_dist'].head(),'\n\n',train['trip_duration'].head())
#print(train.Manhattan_dist.describe())

print(train.Manhattan_dist.max())

#print(test.Manhattan_dist.describe())

print(test.Manhattan_dist.max())
#Feature extraction from Datetime



train['datetime']=pd.to_datetime(train['pickup_datetime'])

train['hour_pick']=train.datetime.dt.hour

train['day_of_week']=train.datetime.dt.dayofweek

train['day_of_month']=train.datetime.dt.days_in_month

train['month']=train.datetime.dt.month

train['is_night_time']=[1 if (i==0 or i>=19)  else 0 for i in train['datetime'].dt.hour]

train['late_night_time']=[1 if (i<5 or i>0)  else 0 for i in train['datetime'].dt.hour]

train['week']=train['datetime'].dt.week

train['min_of_pick']=train['datetime'].dt.minute

train['weather']=[1 if (i in [1,2,3]) else(2 if (i in [4,11,12]) else 3) for i in train['month']]



test['datetime']=pd.to_datetime(test['pickup_datetime'])

test['hour_pick']=test.datetime.dt.hour

test['day_of_week']=test.datetime.dt.dayofweek

test['day_of_month']=test.datetime.dt.days_in_month

test['month']=test.datetime.dt.month

test['is_night_time']=[1 if (i==0 or i>=19)  else 0 for i in test['datetime'].dt.hour]

test['late_night_time']=[1 if (i<5 or i>0)  else 0 for i in test['datetime'].dt.hour]

test['week']=test['datetime'].dt.week

test['min_of_pick']=test['datetime'].dt.minute

test['weather']=[1 if (i in [1,2,3]) else(2 if (i in [4,11,12]) else 3) for i in test['month']]



print(train.shape,test.shape)

print(train.columns,'\n',test.columns)
#Dropping id,pickup_datetime,'dropoff_datetime','pickup_longitude','pickup_latitude',

'dropoff_longitude','dropoff_latitude','x1', 'y1', 'x2','y2','trip_duration',

'datetime',

train=train.drop(['id','pickup_datetime','dropoff_datetime','pickup_longitude','pickup_latitude',

            'dropoff_longitude','dropoff_latitude','x1', 'y1', 'x2','y2','trip_dur_hr','datetime'],1)

print(train.shape,'\n\n',train.columns)
print(train.info())
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

train['store_and_fwd_flag']=le.fit_transform(train['store_and_fwd_flag'])
y=np.log(train['trip_duration'].values + 1)

x=train.drop(['trip_duration'],1)
from sklearn.preprocessing import scale

#x=scale(x)
#print(x.info())
from sklearn.cross_validation import cross_val_score

# Decission Tree regressor

from sklearn.tree import DecisionTreeRegressor

model_dt=DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, 

                               min_samples_split=2, min_samples_leaf=1, 

                               min_weight_fraction_leaf=0.0, max_features=None, 

                               random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, 

                               presort=False)

#print(cross_val_score(model_dt,x,y,cv=5))
#Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

model_rnd_frst=RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, 

                                     min_samples_split=2, min_samples_leaf=1, 

                                     min_weight_fraction_leaf=0.0, max_features='auto', 

                                     max_leaf_nodes=None, min_impurity_split=1e-07, 

                                     bootstrap=True, oob_score=False, n_jobs=-1, 

                                     random_state=None, verbose=1, warm_start=False)

#print(cross_val_score(model_rnd_frst,x,y,cv=2))
from sklearn.ensemble import AdaBoostRegressor

model_ada=AdaBoostRegressor(base_estimator=model_rnd_frst, n_estimators=50, learning_rate=1.0, 

                            loss='linear', random_state=None)

#print(cross_val_score(model_ada,x,y,cv=3))
from sklearn.ensemble import GradientBoostingRegressor

model_gb=GradientBoostingRegressor(loss='ls', learning_rate=0.05, n_estimators=400, subsample=1.0,

                                   criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 

                                   min_weight_fraction_leaf=0.0, max_depth=5, min_impurity_split=1e-07,

                                   init=None, random_state=None, max_features=None, alpha=0.9, 

                                   verbose=0, 

                                   max_leaf_nodes=None, warm_start=False, presort='auto')

#print(cross_val_score(model_gb,x,y,cv=3))
from sklearn.neighbors import KNeighborsRegressor

model_KNN=KNeighborsRegressor(n_neighbors=5, weights='uniform', 

                                                algorithm='auto', leaf_size=30, p=2, 

                                                metric='minkowski', metric_params=None,

                                                n_jobs=-1)

#print(cross_val_score(model_KNN,x,y,cv=3))
from sklearn.svm import SVR

model_svr=SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, 

              shrinking=True, cache_size=800, verbose=False, max_iter=-1)

#print(cross_val_score(model_svr,x,y,cv=2))
import xgboost as xgb

from sklearn.cross_validation import train_test_split

Xtr, Xv, ytr, yv = train_test_split(x, y, test_size=0.2, random_state=1987)

dtrain = xgb.DMatrix(Xtr, label=ytr)

dvalid = xgb.DMatrix(Xv, label=yv)

#dtest = xgb.DMatrix(test[feature_names].values)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



#From beluga's kernel

xgb_pars = {'min_child_weight': 50, 'eta': 0.3, 'colsample_bytree': 0.3, 'max_depth': 10,

            'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 150, watchlist, early_stopping_rounds=100,

                  maximize=False, verbose_eval=10)
#print('Modeling RMSLE %.5f' % model.best_score)
xgb.plot_importance(model, ax=None, height=0.2, xlim=None,

                    ylim=None, title='Feature importance',

                    xlabel='F score', ylabel='Features',

                    importance_type='weight', max_num_features=None,

                    grid=True)

plt.show()

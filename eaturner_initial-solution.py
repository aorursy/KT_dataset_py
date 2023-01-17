# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def toDateTime( df ):

    

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    

    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday

    df['pickup_day'] = df['pickup_datetime'].dt.day

    df['pickup_month'] = df['pickup_datetime'].dt.month

    df['pickup_hour'] = df['pickup_datetime'].dt.hour

    df['pickup_minute'] = df['pickup_datetime'].dt.minute

    df['pickup_dt'] = (df['pickup_datetime'] - df['pickup_datetime'].min()).map(

        lambda x: x.total_seconds())

    

    df.drop('pickup_datetime', axis = 1, inplace = True)



    return df



#get radical distince

def haversine_np(lon1, lat1, lon2, lat2):

   

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2



    c = 2 * np.arcsin(np.sqrt(a))

    km = 6367 * c

    return km



#manhattan distance

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):



    a = haversine_np(lat1, lng1, lat1, lng2)

    b = haversine_np(lat1, lng1, lat2, lng1)

    return a + b



#all distances

def locationFeatures( df ):

    #displacement of degrees

    df['up_town'] = np.sign( df['pickup_longitude'] - df['dropoff_longitude'] )

    df['est_side'] = np.sign( df['pickup_latitude'] - df['dropoff_latitude'] )

     

    #radical distances

    df['haversine_distance'] = haversine_np(

        df['pickup_longitude'], df['pickup_latitude'], 

        df['dropoff_longitude'], df['dropoff_latitude']

    )

    

    #manhattan distances

    df['distance_dummy_manhattan'] = dummy_manhattan_distance(

        df['pickup_latitude'], df['pickup_longitude'],

        df['dropoff_latitude'], df['dropoff_longitude']

    )

    

    df.drop(['pickup_longitude', 'dropoff_longitude'], axis = 1, inplace = True)

    df.drop(['pickup_latitude', 'dropoff_latitude'], axis = 1, inplace = True)

    

    return df



def featureCreate( df ):

    df = toDateTime( df )

    df = locationFeatures( df )

    

    return df
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train = featureCreate( train )

test = featureCreate( test )



#log transform our trip duration

train['trip_duration'] = np.log1p(train['trip_duration'])



#log transform of the haversine distance

train['haversine_distance'] = np.log1p(train['haversine_distance'])

test['haversine_distance'] = np.log1p(test['haversine_distance'])
q1 = np.percentile(train['trip_duration'], 25)

q3 = np.percentile(train['trip_duration'], 75)



iqr = q3 - q1



train = train[ train['trip_duration'] <= q3 + 3.0*iqr]



train = train[ q1 - 3.0*iqr <= train['trip_duration']]
labels = train.pop('trip_duration')

train.drop(["id", "dropoff_datetime"], axis=1, inplace = True)



sub = pd.DataFrame( columns = ['id', 'trip_duration'])

sub['id'] = test.pop('id')
from sklearn.preprocessing import LabelEncoder

for col in train.columns:

    if train[col].dtype == 'object':

        le = LabelEncoder()

        

        le.fit(train[col])

        

        train[col] = le.transform(train[col])

        test[col] = le.transform(test[col])
def modelPredict( est, train, test, labels):

    from sklearn.model_selection import train_test_split

    

    #uncomment for model selection

    

    #x_train, x_val, y_train, y_val = train_test_split(train, labels)

    

    #eval_set = [(x_val, y_val)]

    

    #est.fit(x_train, y_train, eval_set = eval_set, early_stopping_rounds = 100)

    

    print ('Making model...')

    

    est.fit( train, labels )

    

    print ('Done!')

    

    return [est.predict( train ), est.predict( test ) ]
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



#these were cross-validated using the current features and some intuition

xgb = XGBRegressor(max_depth = 7, learning_rate = 1e-1, n_estimators = 479,

                      subsample = 0.8, n_jobs = 4)



lgbm = LGBMRegressor(max_depth = 7, learning_rate = 1e-1, n_estimators = 582,

                      subsample = 0.8, nthread = 4)
train_preds = pd.DataFrame( columns = ['xgb_pred', 'lgbm_pred'] )

test_preds = pd.DataFrame( columns = ['xgb_pred', 'lgbm_pred'] )
preds = modelPredict(xgb, train, test, labels)



train_preds['xgb_pred'] = preds[0]

test_preds['xgb_pred'] = preds[1]



print ('On to the next one')



preds = modelPredict(lgbm, train, test, labels)



train_preds['lgbm_pred'] = preds[0]

test_preds['lgbm_pred'] = preds[1]



print ('Finished')
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



ax.scatter(train_preds['xgb_pred'], train_preds['lgbm_pred'], labels)



plt.show()
fig = plt.figure()

ax = fig.add_subplot(111)



ax.scatter(train_preds['xgb_pred'], labels, c='b', marker="s", label='xgb')

ax.scatter(train_preds['lgbm_pred'], labels, c='r', marker="o", label='lbgm')



plt.legend()



plt.show()
fig = plt.figure()

ax = fig.add_subplot(111)



res = train_preds['xgb_pred'].values - labels.values

ax.scatter(train_preds['xgb_pred'], res, c='b', marker="s", label='xgb_res')



res = train_preds['lgbm_pred'].values - labels.values

ax.scatter(train_preds['lgbm_pred'], res, c='r', marker="o", label='lbgm_res')



plt.legend()



plt.show()
preds = test_preds.mean(axis=1)
sub['trip_duration'] = np.expm1(preds)

sub.to_csv('submission.csv', index=False)
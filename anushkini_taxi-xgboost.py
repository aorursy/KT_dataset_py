# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.|

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#package imports

import pandas as pd

from sklearn import preprocessing

import xgboost as xgb

from sklearn.metrics import mean_squared_error

from bayes_opt import BayesianOptimization

from sklearn.model_selection import train_test_split

import gc

import os

from tqdm import tqdm

import holidays

import datetime as dt

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline
#read the test and train sets

gc.collect()

df_chunk = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = 2_000_000)

test_df = pd.read_feather('../input/kernel318ff03a29/test_feature.feather')

gc.collect()
#examine the dataset's fist 5 rows

df_chunk.head()
#examine the dataset's fist 5 rows

df_chunk.head()
us_holidays = holidays.US()
df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)

df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')

us_holidays = holidays.US()

def haversine_distance(lat1, long1, lat2, long2):

    R = 6371  #radius of earth in kilometers

    phi1 = np.radians(lat1)

    phi2 = np.radians(lat2)

    delta_phi = np.radians(lat2-lat1)

    delta_lambda = np.radians(long2-long1)

    #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2

    #c = 2 * atan2( √a, √(1−a) )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    #d = R*c

    d = (R * c) #in kilometers

    return d



df_chunk["time"] = pd.to_numeric(df_chunk.apply(lambda r: r.pickup_datetime.hour*60 + r.pickup_datetime.minute, axis = 1), downcast = "unsigned")

gc.collect()

df_chunk["holiday"] = pd.to_numeric(df_chunk.apply(lambda x: 1 if x.pickup_datetime.strftime('%d-%m-%y')in us_holidays else 0, axis =1), downcast = "unsigned")

gc.collect()

Manhattan = (-73.9712,40.7831)[::-1]

JFK_airport = (-73.7781,40.6413)[::-1]

Laguardia_airport = (-73.8740,40.7769)[::-1]

statue_of_liberty = (-74.0445,40.6892)[::-1]

central_park = (-73.9654,40.7829)[::-1]

time_square = (-73.9855,40.7580)[::-1]

brooklyn_bridge = (-73.9969,40.7061)[::-1]

rockerfeller = (-73.9787,40.7587)[::-1]



df_chunk["distance"] = pd.to_numeric(haversine_distance(df_chunk['pickup_latitude'], df_chunk['pickup_longitude'], df_chunk['dropoff_latitude'], df_chunk['dropoff_longitude']), downcast = 'float')

df_chunk["year"] = df_chunk["pickup_datetime"].dt.year

df_chunk["weekday"] = pd.to_numeric(df_chunk["pickup_datetime"].dt.weekday, downcast= "unsigned")



df_chunk['pickup_distance_Mtn'] = pd.to_numeric(haversine_distance(Manhattan[0],Manhattan[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

df_chunk['dropoff_distance_Mtn'] = pd.to_numeric(haversine_distance(Manhattan[0],Manhattan[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')

df_chunk['dropoff_distance_jfk'] = pd.to_numeric(haversine_distance(JFK_airport[0],JFK_airport[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')

df_chunk['pickup_distance_jfk'] = pd.to_numeric(haversine_distance(JFK_airport[0],JFK_airport[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

df_chunk['pickup_distance_lg'] = pd.to_numeric(haversine_distance(Laguardia_airport[0],Laguardia_airport[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')

df_chunk['dropoff_distance_lg'] = pd.to_numeric(haversine_distance(Laguardia_airport[0],Laguardia_airport[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')



df_chunk['day'] = df_chunk['pickup_datetime'].dt.day

df_chunk['month'] = df_chunk['pickup_datetime'].dt.month



test_df['day'] = test_df['pickup_datetime'].dt.day

test_df['month'] = test_df['pickup_datetime'].dt.month



df_chunk['pickup_distance_sol'] = pd.to_numeric(haversine_distance(statue_of_liberty[0],statue_of_liberty[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

df_chunk['dropoff_distance_sol'] = pd.to_numeric(haversine_distance(statue_of_liberty[0],statue_of_liberty[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')

df_chunk['pickup_distance_cp'] = pd.to_numeric(haversine_distance(central_park[0],central_park[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

df_chunk['dropoff_distance_cp'] = pd.to_numeric(haversine_distance(central_park[0],central_park[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')

df_chunk['pickup_distance_ts'] = pd.to_numeric(haversine_distance(time_square[0],time_square[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

df_chunk['dropoff_distance_ts'] = pd.to_numeric(haversine_distance(time_square[0],time_square[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')

df_chunk['pickup_distance_bb'] = pd.to_numeric(haversine_distance(brooklyn_bridge[0],brooklyn_bridge[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

df_chunk['dropoff_distance_bb'] = pd.to_numeric(haversine_distance(brooklyn_bridge[0],brooklyn_bridge[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')

df_chunk['pickup_distance_r'] = pd.to_numeric(haversine_distance(rockerfeller[0],rockerfeller[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

df_chunk['dropoff_distance_r'] = pd.to_numeric(haversine_distance(rockerfeller[0],rockerfeller[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')



test_df['pickup_distance_sol'] = pd.to_numeric(haversine_distance(statue_of_liberty[0],statue_of_liberty[1],test_df['pickup_latitude'],test_df['pickup_longitude']), downcast = 'float')

test_df['dropoff_distance_sol'] = pd.to_numeric(haversine_distance(statue_of_liberty[0],statue_of_liberty[1],test_df['dropoff_latitude'],test_df['dropoff_longitude']), downcast = 'float')

test_df['pickup_distance_cp'] = pd.to_numeric(haversine_distance(central_park[0],central_park[1],test_df['pickup_latitude'],test_df['pickup_longitude']), downcast = 'float')

test_df['dropoff_distance_cp'] = pd.to_numeric(haversine_distance(central_park[0],central_park[1],test_df['dropoff_latitude'],test_df['dropoff_longitude']), downcast = 'float')

test_df['pickup_distance_ts'] = pd.to_numeric(haversine_distance(time_square[0],time_square[1],test_df['pickup_latitude'],test_df['pickup_longitude']), downcast = 'float')

test_df['dropoff_distance_ts'] = pd.to_numeric(haversine_distance(time_square[0],time_square[1],test_df['dropoff_latitude'],test_df['dropoff_longitude']), downcast = 'float')

test_df['pickup_distance_bb'] = pd.to_numeric(haversine_distance(brooklyn_bridge[0],brooklyn_bridge[1],test_df['pickup_latitude'],test_df['pickup_longitude']), downcast = 'float')

test_df['dropoff_distance_bb'] = pd.to_numeric(haversine_distance(brooklyn_bridge[0],brooklyn_bridge[1],test_df['dropoff_latitude'],test_df['dropoff_longitude']), downcast = 'float')

test_df['pickup_distance_r'] = pd.to_numeric(haversine_distance(rockerfeller[0],rockerfeller[1],test_df['pickup_latitude'],test_df['pickup_longitude']), downcast = 'float')

test_df['dropoff_distance_r'] = pd.to_numeric(haversine_distance(rockerfeller[0],rockerfeller[1],test_df['dropoff_latitude'],test_df['dropoff_longitude']), downcast = 'float')



df_chunk['pickup_longitude'] = np.radians(df_chunk['pickup_longitude'])

df_chunk['pickup_latitude'] = np.radians(df_chunk['pickup_latitude'])

df_chunk['dropoff_latitude'] = np.radians(df_chunk['dropoff_latitude'])

df_chunk['dropoff_longitude'] = np.radians(df_chunk['dropoff_longitude'])



test_df['pickup_longitude'] = np.radians(test_df['pickup_longitude'])

test_df['pickup_latitude'] = np.radians(test_df['pickup_latitude'])

test_df['dropoff_latitude'] = np.radians(test_df['dropoff_latitude'])

test_df['dropoff_longitude'] = np.radians(test_df['dropoff_longitude'])
df_chunk = df_chunk.drop(['pickup_datetime'],axis = 1)

df_chunk = df_chunk.drop(['key'],axis = 1)

X_train, X_test, y_train, y_test = train_test_split(df_chunk.drop('fare_amount', axis=1),

                                                    df_chunk['fare_amount'], test_size=0.1)

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test)

del(df_chunk)

del(X_train)

del(X_test)

gc.collect()
def xgb_evaluate(max_depth, gamma, colsample_bytree):

    params = {'eval_metric': 'rmse',

              'max_depth': int(max_depth),

              'subsample': 0.8,

              'eta': 0.1,

              'gamma': gamma,

              'colsample_bytree': colsample_bytree}

    cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    

    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]
xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (7, 12), 

                                             'gamma': (0, 1),

                                             'colsample_bytree': (0.5, 0.9)})

xgb_bo.maximize(init_points=5, n_iter=10, acq='ei')
sorted_res = sorted(xgb_bo.res,key = lambda x: x['target'])

params = sorted_res[-1]

params['params']['max_depth'] = int(params['params']['max_depth']) 
model = xgb.train(params, dtrain, num_boost_round=1000)



# Predict on testing and training set

y_pred = model.predict(dtest)

y_train_pred = model.predict(dtrain)



# Report testing and training RMSE

print(np.sqrt(mean_squared_error(y_test, y_pred)))

print(np.sqrt(mean_squared_error(y_train, y_train_pred)))
#Feature Importance

fscores = pd.DataFrame({'X': list(model.get_fscore().keys()), 'Y': list(model.get_fscore().values())})

fscores.sort_values(by='Y').plot.bar(x='X')
test_df.head()
X_test = test_df.iloc[:,2:]

X_test = xgb.DMatrix(X_test)

pred = model.predict(X_test)
#create a dataframe in the submission format

holdout = pd.DataFrame({'key': test_df.key, 'fare_amount': pred})

#write the submission file to output

holdout.to_csv('submission.csv', index=False)
holdout.head()
len(holdout)
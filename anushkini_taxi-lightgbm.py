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
!pip install holidays



#package imports

import pandas as pd

from sklearn import preprocessing

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from bayes_opt import BayesianOptimization

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

import seaborn as sns

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

df_chunk = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = 15_000_000)

test_df = pd.read_feather('../input/data-feature-engineering/test_feature.feather')



gc.collect()
#examine the dataset's fist 5 rows

df_chunk.head()
len(df_chunk)
df_chunk = df_chunk.dropna()

df_chunk = df_chunk[(df_chunk['fare_amount'] > 0) & (df_chunk['fare_amount'] <= 250) & 

          (df_chunk['passenger_count'] >= 0) & (df_chunk['passenger_count'] <= 8)  & 

          ((df_chunk['pickup_longitude'] != 0) & (df_chunk['pickup_latitude'] != 0) & (df_chunk['dropoff_longitude'] != 0) & (df_chunk['dropoff_latitude'] != 0))]
len(df_chunk)
#examine the dataset's fist 5 rows

test_df.head()
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
df_chunk.head()
test_df.head()
y = df_chunk['fare_amount']

df_chunk = df_chunk.drop(['key','pickup_datetime','fare_amount'],axis = 1)

X_train,X_val,y_train,y_val = train_test_split(df_chunk,y,test_size = 0.1)

del(df_chunk)

del(y)

gc.collect()
dtrain = lgb.Dataset(X_train,y_train,silent=False,categorical_feature=['year','month','day','weekday'])

dval = lgb.Dataset(X_val,y_val,silent=False,categorical_feature=['year','month','day','weekday'])
lgbm_params =  {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'rmse',

    'nthread': 4,

    'learning_rate': 0.05,

    'bagging_fraction': 1,

    'num_rounds':50000

    }

model = lgb.train(lgbm_params, train_set = dtrain, num_boost_round=10000,early_stopping_rounds=500,verbose_eval=500, valid_sets=dval)

del(X_train)

del(y_train)

del(X_val)

del(y_val)

gc.collect()



#reference - https://stackoverflow.com/questions/55208734/save-lgbmregressor-model-from-python-lightgbm-package-to-disc

model.save_model("model.txt")
"""

#Reference: 

trainshape = X_train.shape

testshape = X_val.shape



print("Light Gradient Boosting Regressor: ")





folds = KFold(n_splits=5, shuffle=True)

fold_preds = np.zeros(testshape[0])

oof_preds = np.zeros(trainshape[0])

dtrain.construct()



# Fit 5 Folds

for trn_idx, val_idx in folds.split(X_train):

    clf = lgb.train(

        params=lgbm_params,

        train_set=dtrain.subset(trn_idx),

        valid_sets=dtrain.subset(val_idx),

        num_boost_round=20000, 

        early_stopping_rounds=1000,

        verbose_eval=500

    )

    oof_preds[val_idx] = clf.predict(dtrain.data.iloc[val_idx])

    fold_preds += clf.predict(X_val) / folds.n_splits

    print(mean_squared_error(y_train.iloc[val_idx], oof_preds[val_idx]) ** .5)

"""
test_key = test_df['key']

test_df = test_df.drop(['key','pickup_datetime'],axis = 1)

pred = model.predict(test_df)
#create a dataframe in the submission format

holdout = pd.DataFrame({'key': test_key, 'fare_amount': pred})

#write the submission file to output

holdout.to_csv('submission.csv', index=False)
holdout.head()
feature_imp = pd.DataFrame({'Value':model.feature_importance(),'Feature':test_df.columns})



plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.show()
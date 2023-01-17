# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'

# Assume we only know that the csv file is somehow large, but not the exact size
# we want to know the exact number of rows

# Method 1, using file.readlines. Takes about 20 seconds.
#with open(TRAIN_PATH) as file:
    #n_rows = len(file.readlines())

#print (f'Exact number of rows: {n_rows}')
# Peep at the training file header
#df_tmp = pd.read_csv(TRAIN_PATH, nrows=5)
#df_tmp.head()
#df_tmp.info()
# Set columns to most suitable type to optimize for memory usage
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())
NROWS = 12000000
test_df = pd.read_csv(TEST_PATH)
train_df = pd.read_csv(TRAIN_PATH, usecols=cols,nrows = NROWS, dtype=traintypes)
#NROWS = 6000000
#chunksize = 5_000_000 # 5 million rows at one go. Or try 10 million
#total_chunk = NROWS // chunksize + 1
#print(f'Chunk size: {chunksize:,}\nTotal chunks required: {total_chunk}')
'''
df_list = [] # list to hold the batch dataframe
i=0

for df_chunk in pd.read_csv(TRAIN_PATH, usecols=cols,nrows = NROWS, dtype=traintypes, chunksize=chunksize):
    
    i = i+1
    # Each chunk is a corresponding dataframe
    print(f'DataFrame Chunk {i:02d}/{total_chunk}')
    
    # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
    # Using parse_dates would be much slower!
    df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
    df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    
    # Can process each chunk of dataframe here
    # clean_data(), feature_engineer(),fit()
    
    # Alternatively, append the chunk to list and merge all
    df_list.append(df_chunk) 
    
    # Merge all dataframes into one dataframe
train_df = pd.concat(df_list)

# Delete the dataframe list to release memory
del df_list

# See what we have loaded
train_df.info()
'''
#display(train_df.head())
#display(train_df.tail())
#Identify null values
#print(train_df.isnull().sum())
#Drop rows with null values
train_df = train_df.dropna(how = 'any', axis = 'rows')
#Plot variables using only 1000 rows for efficiency
#train_df.iloc[:1000].plot.scatter('pickup_longitude', 'pickup_latitude')
#train_df.iloc[:1000].plot.scatter('dropoff_longitude', 'dropoff_latitude')
#Clean dataset
def clean_df(df):
    return df[(df.fare_amount > 0) & 
            (df.pickup_longitude > -80) & (df.pickup_longitude < -70) &
            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &
            (df.dropoff_longitude > -80) & (df.dropoff_longitude < -70) &
            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45) &
            (df.passenger_count > 0) & (df.passenger_count < 10)]

#train_df = clean_df(train_df)
#print(len(train_df))
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
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

def add_airport_dist(dataset):
    """
    Return minumum distance from pickup or dropoff coordinates to each airport.
    JFK: John F. Kennedy International Airport
    EWR: Newark Liberty International Airport
    LGA: LaGuardia Airport
    """
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    
    pickup_lat = dataset['pickup_latitude']
    dropoff_lat = dataset['dropoff_latitude']
    pickup_lon = dataset['pickup_longitude']
    dropoff_lon = dataset['dropoff_longitude']
    
    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon) 
    
    dataset['jfk_dist'] = pd.concat([pickup_jfk, dropoff_jfk], axis=1).min(axis=1)
    dataset['ewr_dist'] = pd.concat([pickup_ewr, dropoff_ewr], axis=1).min(axis=1)
    dataset['lga_dist'] = pd.concat([pickup_lga, dropoff_lga], axis=1).min(axis=1)
    
    return dataset
    
def add_datetime_info(dataset):
    #Convert to datetime format
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")
    
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['weekday'] = dataset.pickup_datetime.dt.weekday
    dataset['year'] = dataset.pickup_datetime.dt.year
    
    return dataset

#train_df = add_datetime_info(train_df)
#train_df = add_airport_dist(train_df)
#train_df['distance'] = sphere_dist(train_df['pickup_latitude'], train_df['pickup_longitude'], 
                                   #train_df['dropoff_latitude'] , train_df['dropoff_longitude'])

#train_df.head()
#train_df.drop(columns=['key', 'pickup_datetime'], inplace=True)
#train_df.head()
def transform_features(df):
    df = add_datetime_info(df)
    df = add_airport_dist(df)
    df['distance'] = sphere_dist(df['pickup_latitude'], df['pickup_longitude'], 
                                   df['dropoff_latitude'] , df['dropoff_longitude'])
    df.drop(columns=['pickup_datetime'], inplace=True)
    return df 
train_df = transform_features(train_df)
test_df = transform_features(test_df)
test_df.drop(columns=['key'], inplace=True)
#train_df.head()
#test_df.columns
train_df_y = train_df.fare_amount.copy()
train_df_X = train_df[test_df.columns]
#train_df_X.head()

#test_df.head()
#print("Does Train feature equal test feature?: ", all(train_df_X.columns == test_df.columns))
trainshape = train_df_X.shape
testshape = test_df.shape
# LGBM Dataset Formating
dtrain = lgb.Dataset(train_df_X, label = train_df_y, free_raw_data = False)
print("Light Gradient Boosting Regressor: ")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate' :'0.03',
    'num_leaves':'31',
    'max_depth' : '-1',
    'subsample' :'.8',
     'colsample_bytree' : '0.6',
        'min_split_gain' : '0.5',
        'min_child_weight' : '1',
        'min_child_samples' :'10',
        'scale_pos_weight' : '1',
        'num_threads' : '4',
        'seed' : '0',
        'eval_freq' : '50'
                }

folds = KFold(n_splits=5, shuffle=True, random_state=1)
fold_preds = np.zeros(testshape[0])
oof_preds = np.zeros(trainshape[0])
dtrain.construct()

# Fit 5 Folds
modelstart = time.time()
for trn_idx, val_idx in folds.split(train_df_X):
    clf = lgb.train(
        params=lgbm_params,
        train_set=dtrain.subset(trn_idx),
        valid_sets=dtrain.subset(val_idx),
        num_boost_round=10000, 
        early_stopping_rounds=125,
        verbose_eval=500     
    )
    oof_preds[val_idx] = clf.predict(dtrain.data.iloc[val_idx])
    fold_preds += clf.predict(test_df) / folds.n_splits
    print(mean_squared_error(train_df_y.iloc[val_idx], oof_preds[val_idx]) ** .5)
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
test_df = pd.read_csv(TEST_PATH)
#result = pd.DataFrame(fold_preds,columns=["fare_amount"],index=testdex)
result = pd.DataFrame({'key':test_df['key'], 'fare_amount':fold_preds})
result.head()
result.to_csv('taxi-fare-prediction.csv', index=False)
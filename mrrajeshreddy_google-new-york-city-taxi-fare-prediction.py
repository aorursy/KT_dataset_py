import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from datetime import datetime
%matplotlib inline
#tf.enable_eager_execution()
TRAIN_FILE_NAME = '../input/train.csv'
TEST_FILE_NAME = '../input/test.csv'
NUM_RECORDS = 6000000
datatypes = {'key': 'str', 
              'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'int',
            }
df = pd.read_csv(TRAIN_FILE_NAME, 
                 nrows=NUM_RECORDS, 
                 dtype=datatypes, 
                 parse_dates=['pickup_datetime'])
len(df)
lat_border = (40.63, 40.85)
long_border = (-74.03, -73.75)
df.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude', 
       color='green', s=.02, alpha=.6)
plt.title("Dropoffs")
plt.ylim(lat_border)
plt.xlim(long_border)

df.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', 
       color='blue', s=.02, alpha=.6)
plt.title("Pickups")
plt.ylim(lat_border)
plt.xlim(long_border)
plt.show()
df.head()
df.isnull().sum()
print(f'Count before:{len(df)}')
df = df.dropna(how='any', axis=0)
print(f'Count after:{len(df)}')
df.describe()
df[df.passenger_count == 0].shape
df[['fare_amount','passenger_count']].corr()
_ = df['passenger_count'].hist(bins=10, figsize = (8,4) )
df[df.fare_amount < 0].shape
df[df.fare_amount > 250].shape
_ = df[df.fare_amount < 250]['fare_amount'].hist(bins=10, figsize = (8,4) )
def clean(df):
    df = df[(-76 <= df['pickup_longitude']) & (df['pickup_longitude'] <= -72)]
    df = df[(-76 <= df['dropoff_longitude']) & (df['dropoff_longitude'] <= -72)]
    df = df[(38 <= df['pickup_latitude']) & (df['pickup_latitude'] <= 42)]
    df = df[(38 <= df['dropoff_latitude']) & (df['dropoff_latitude'] <= 42)]
    # Remove possible outliers
    df = df[(0 < df['fare_amount']) & (df['fare_amount'] <= 250)]
    # Remove inconsistent values
    df = df[(df['dropoff_longitude'] != df['pickup_longitude'])]
    df = df[(df['dropoff_latitude'] != df['pickup_latitude'])]
    return df
print(f'Count before cleaning:{len(df)}')
df = clean(df)
print(f'Count after cleaning:{len(df)}')
def check_night(row):
    hour = row['hour']
    hclass = 5
    if hour > 6 and hour <= 11:
        hclass = 1
    elif hour >11 and hour <= 16:
        hclass = 2
    elif hour > 16 and hour <= 20:
        hclass = 3
    elif hour > 20 and hour <= 23:
        hclass = 4
    else:
        hclass = 5
    return hclass
def process_date(df):
    df['year'] = df.pickup_datetime.apply(lambda x: x.year)
    df['month'] = df.pickup_datetime.apply(lambda x: x.month)
    df['day'] = df.pickup_datetime.apply(lambda x: x.day)
    df['hour'] = df.pickup_datetime.apply(lambda x: x.hour)
    df['weekday'] = df.pickup_datetime.apply(lambda x: x.weekday())
    df['hclass'] = df.apply (lambda x: check_night(x), axis=1)
    return df
boroughs = {
    'manhattan':{
        'min_lon':-74.0479, 'min_lat':40.6829,
        'max_lon':-73.9067, 'max_lat':40.8820
    },
    'queens':{
        'min_lon':-73.9630, 'min_lat':40.5431,
        'max_lon':-73.7004, 'max_lat':40.8007
    },
    'brooklyn':{
        'min_lon':-74.0421, 'min_lat':40.5707,
        'max_lon':-73.8334, 'max_lat':40.7395
    },
    'bronx':{
        'min_lon':-73.9339, 'min_lat':40.7855,
        'max_lon':-73.7654, 'max_lat':40.9176
    },
    'staten_island':{
        'min_lon':-74.2558, 'min_lat':40.4960,
        'max_lon':-74.0522, 'max_lat':40.6490
    }
}
airports = {
    'JFK':{
        'min_lon':-73.8352, 'min_lat':40.6195,
        'max_lon':-73.7401, 'max_lat':40.6659
    },          
    'EWR':{
        'min_lon':-74.1925, 'min_lat':40.6700, 
        'max_lon':-74.1531, 'max_lat':40.7081
    },
    'LaGuardia':{
        'min_lon':-73.8895, 'min_lat':40.7664, 
        'max_lon':-73.8550, 'max_lat':40.7931
    }
}
boroughs_list = list(boroughs.keys())
boroughs_list.append('others')
airport_list = list(airports.keys())
airport_list.append('others')
print(boroughs_list)
print(airport_list)
def getBorough(lat,lon):
    locs=boroughs.keys()
    for loc in locs:
        if lat>=boroughs[loc]['min_lat'] and lat<=boroughs[loc]['max_lat'] and lon>=boroughs[loc]['min_lon'] and lon<=boroughs[loc]['max_lon']:
            return loc
    return 'others'
def getAirport(lat,lon):
    locs=airports.keys()
    for loc in locs:
        if lat>=airports[loc]['min_lat'] and lat<=airports[loc]['max_lat'] and lon>=airports[loc]['min_lon'] and lon<=airports[loc]['max_lon']:
            return loc
    return 'others'
def process_distance(df):
    lat1 = df.pickup_latitude
    lat2 = df.dropoff_latitude
    lon1 = df.pickup_longitude
    lon2 = df.dropoff_longitude
    df['pickup_borough'] = df.apply(lambda row:getBorough(row['pickup_latitude'],
                                                          row['pickup_longitude']),
                                     axis=1)
    df['dropoff_borough'] = df.apply(lambda row:getBorough(row['dropoff_latitude'],
                                                           row['dropoff_longitude']),
                                     axis=1)
    df['pickup_airport'] = df.apply(lambda row:getAirport(row['pickup_latitude'],
                                                          row['pickup_longitude']),
                                     axis=1)
    df['dropoff_airport'] = df.apply(lambda row:getAirport(row['dropoff_latitude'],
                                                           row['dropoff_longitude']),
                                     axis=1)
    df['latdiff'] = lat1 - lat2
    df['londiff'] = lon1 - lon2
    df['euclid'] = np.sqrt((df['latdiff'] * df['latdiff']) + (df['londiff'] * df['londiff']) )
    df['manhattan'] =  np.abs(lat2 - lat1) + np.abs(lon2 - lon1)
    p = np.pi / 180
    d =  0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    df['haversine'] = 0.6213712 * 12742 * np.arcsin(np.sqrt(d))
    return df
def process(df):
    df = process_date(df)
    df = process_distance(df)
    return df
    
df = process(df)
df.head()
df[df.hclass < 1].shape
df[['hour','month','day','hclass','euclid','manhattan', 'haversine']].describe()
corr_matrix = df[['fare_amount','euclid','manhattan', 'haversine']].corr()
_ = sns.heatmap(corr_matrix)
train_df, validation_df = train_test_split(df, test_size=0.1, random_state=1, shuffle=True)
print('Training examples:', len(train_df))
print('Validation examples:', len(validation_df))
scale_cols = ['pickup_longitude', 'pickup_latitude','dropoff_longitude', 
              'dropoff_latitude','year', 'latdiff', 'londiff',
              'euclid', 'manhattan', 'haversine']
scaler = preprocessing.MinMaxScaler()
train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
validation_df[scale_cols] = scaler.transform(validation_df[scale_cols])
train_df.head()
validation_df.head()
train_df.describe()
def get_required_data_df(df, include_label=False):
    required_cols = ['pickup_longitude', 
                     'pickup_latitude',
                     'dropoff_longitude',
                     'dropoff_latitude',
                     'month',
                     'day',
                     'hour',
                     'weekday',
                     'hclass',
                     'latdiff',
                     'londiff',
                     'euclid',
                     'manhattan',
                     'haversine',
                     'pickup_borough',
                     'dropoff_borough',
                     'pickup_airport',
                     'dropoff_airport',
                    ]
    if include_label:
        required_cols.append('fare_amount')
    data_df = df[required_cols]
    return(data_df)
validation_data_df = get_required_data_df(validation_df, include_label=True)
validation_data_df.head()
train_data_df = get_required_data_df(train_df, include_label=True)
train_data_df.head()
INPUT_COLUMNS = [
    # Define features
    tf.feature_column.categorical_column_with_identity('month', num_buckets = 13),
    tf.feature_column.categorical_column_with_identity('day', num_buckets = 32),
    tf.feature_column.categorical_column_with_identity('weekday', num_buckets = 7),
    tf.feature_column.categorical_column_with_identity('hour', num_buckets = 24),
    tf.feature_column.categorical_column_with_identity('hclass', num_buckets = 6),
    tf.feature_column.categorical_column_with_vocabulary_list('pickup_borough', 
                                                              vocabulary_list=boroughs_list),
    tf.feature_column.categorical_column_with_vocabulary_list('dropoff_borough', 
                                                              vocabulary_list=boroughs_list),
    tf.feature_column.categorical_column_with_vocabulary_list('pickup_airport', 
                                                              vocabulary_list=airport_list),
    tf.feature_column.categorical_column_with_vocabulary_list('dropoff_airport', 
                                                              vocabulary_list=airport_list),

    # Distance columns
    tf.feature_column.numeric_column('pickup_latitude'),
    tf.feature_column.numeric_column('pickup_longitude'),
    tf.feature_column.numeric_column('dropoff_latitude'),
    tf.feature_column.numeric_column('dropoff_longitude'),
    tf.feature_column.numeric_column('latdiff'),
    tf.feature_column.numeric_column('londiff'),
    tf.feature_column.numeric_column('euclid'),
    tf.feature_column.numeric_column('manhattan'),
    tf.feature_column.numeric_column('haversine'),
]
def build_estimator(input_columns, 
                    nbuckets,
                    hidden_units,
                    linear_optimizer='Ftrl',
                    dnn_optimizer = 'Adagrad',
                    run_config=None
                   ):
    ( month, day, dayofweek, 
     hourofday, hclass, 
      pbor, dbor, pair, dair,
     plat, plon, dlat, dlon, latdiff, londiff, 
     euclidean ,manhattan, haversine, 
    ) = input_columns
    
     # Bucketize the lats & lons
    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-76.0, -72.0, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)
    
    # Feature cross
    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets * nbuckets)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets * nbuckets)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4 )
    day_hr =  tf.feature_column.crossed_column([dayofweek, hourofday], 24 * 7)
    
    # Wide columns
    wide_columns = [
        # Feature crosses
        dloc, ploc, pd_pair,
        pbor, dbor, pair, dair,
        day_hr,
        # Sparse columns
        month, day, dayofweek, hourofday, hclass,
    ]
    
    # deep columns
    deep_columns = [
        # Embedding_column to "group" together ...
        tf.feature_column.embedding_column(pd_pair, 10),
        tf.feature_column.embedding_column(day_hr, 10),
        tf.feature_column.embedding_column(pbor, 10),
        tf.feature_column.embedding_column(dbor, 10),
        tf.feature_column.embedding_column(pair, 10),
        tf.feature_column.embedding_column(dair, 10),
  
        # Numeric columns
        plat, plon, dlat, dlon,
        latdiff, londiff, euclidean, manhattan, haversine
    ]
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        #model_dir = model_dir,
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_hidden_units = hidden_units,
        dnn_optimizer = dnn_optimizer,
        linear_optimizer = linear_optimizer,
        loss_reduction = tf.losses.Reduction.MEAN
    )
    
    return estimator
BATCH_SIZE = 512
MAX_STEPS = 200000
NBUCKETS = 10
LEARNING_RATE = 0.0001
HIDDEN_UNITS = [64, 64, 64, 4]
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
estimator = build_estimator(INPUT_COLUMNS, 
                            NBUCKETS, 
                            HIDDEN_UNITS, 
                            dnn_optimizer=optimizer)
def pandas_input_fun(df, batch_size=None, mode=tf.estimator.ModeKeys.TRAIN):
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        label = df['fare_amount']
        return tf.estimator.inputs.pandas_input_fn(x = df, 
                                                   y = label, 
                                                   batch_size=batch_size, 
                                                   num_epochs=100,
                                                   shuffle=True
                                                )
    elif mode == tf.estimator.ModeKeys.EVAL:
        label = df['fare_amount']
        return tf.estimator.inputs.pandas_input_fn(x = df, 
                                                   y = label, 
                                                   batch_size=batch_size, 
                                                   num_epochs=100
                                                )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.inputs.pandas_input_fn(x = df, 
                                                   y = None, 
                                                   num_epochs=1,
                                                   shuffle=False
                                                )
train_spec = tf.estimator.TrainSpec(input_fn=pandas_input_fun(train_data_df, 
                                                              BATCH_SIZE, 
                                                              tf.estimator.ModeKeys.TRAIN), 
                                    max_steps=MAX_STEPS
                                   )
eval_spec = tf.estimator.EvalSpec(input_fn=pandas_input_fun(validation_data_df, 
                                                             BATCH_SIZE, 
                                                             tf.estimator.ModeKeys.EVAL), 
                                  steps=500
                                 )
tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)
predictions = estimator.predict(input_fn=pandas_input_fun(validation_data_df, 
                                                          None, 
                                                          tf.estimator.ModeKeys.PREDICT)
                               )
predictions_pd = pd.DataFrame(predictions)
metrics.mean_squared_error(validation_data_df.fare_amount, predictions_pd.predictions)
plt.scatter(validation_data_df.euclid,validation_data_df.fare_amount, color='red')
plt.scatter(validation_data_df.euclid,predictions_pd.predictions, color='blue')
#plt.plot(validation_data_df.euclid, predictions_pd.predictions, color='blue')
_ = plt.show()
test_df = pd.read_csv(TEST_FILE_NAME, dtype=datatypes, parse_dates=['pickup_datetime'])
test_df = process(test_df)
#test_df = process_date(test_df)
#test_df = process_distance(test_df)
test_df[scale_cols] = scaler.transform(test_df[scale_cols])
test_df.head()
test_data_df = get_required_data_df(test_df)
test_data_df.head()
predictions = estimator.predict(input_fn=pandas_input_fun(test_data_df, 
                                                          None, 
                                                          tf.estimator.ModeKeys.PREDICT)
                               )
predictions_df = pd.DataFrame(predictions)
predictions_df = pd.DataFrame(predictions_df['predictions'].apply(lambda x: x[0]))
submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': predictions_df.predictions},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)
submission.head()
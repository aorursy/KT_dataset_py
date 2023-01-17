import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
datatypes = {'key': 'str', 
              'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

train_df = pd.read_csv('../input/train.csv', nrows=5000000, dtype=datatypes)
train_df.describe()
test_df = pd.read_csv('../input/test.csv', dtype=datatypes)
test_df.describe()
def manhattan_distance(lat1, long1, lat2, long2):
    diff_lat = abs(lat1 - lat2)
    diff_long = abs(long1 - long2)
    return (diff_lat + diff_long)

def distance_between_points(df):
    df['diff_lat'] = abs(df['dropoff_latitude'] - df['pickup_latitude'])
    df['diff_long'] = abs(df['dropoff_longitude'] - df['pickup_longitude'])
    df['manhattan_dist'] = df['diff_lat'] + df['diff_long']
    
    jfk = [40.6413, -73.7781]
    lga = [40.7769, -73.8740]
    ewr = [40.6895, -74.1745]
    # how far was this ride from the 3 nearby airports?
    df['jfk_dist_pickup'] = manhattan_distance(df['pickup_latitude'], df['pickup_longitude'], jfk[0], jfk[1])
    df['jfk_dist_dropoff'] = manhattan_distance(df['dropoff_latitude'], df['dropoff_longitude'], jfk[0], jfk[1])
    df['lga_dist_pickup'] = manhattan_distance(df['pickup_latitude'], df['pickup_longitude'], lga[0], lga[1])
    df['lga_dist_dropoff'] = manhattan_distance(df['dropoff_latitude'], df['dropoff_longitude'], lga[0], lga[1])
    df['ewr_dist_pickup'] = manhattan_distance(df['pickup_latitude'], df['pickup_longitude'], ewr[0], ewr[1])
    df['ewr_dist_dropoff'] = manhattan_distance(df['dropoff_latitude'], df['dropoff_longitude'], ewr[0], ewr[1])
    
distance_between_points(train_df)
def extract_date_details(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')
    df['year'] = df['pickup_datetime'].apply(lambda date: date.year)
    df['month'] = df['pickup_datetime'].apply(lambda date: date.month)
    df['day'] = df['pickup_datetime'].apply(lambda date: date.weekday())
    df['hour'] = df['pickup_datetime'].apply(lambda date: date.hour)
    
extract_date_details(train_df)
train_df
def remove_outliers(df):
    # remove nulls
    df = df.dropna()
    
    # remove any lat/long changes that are too big or too small
    df = df[(df['diff_lat'] < 5.0) & (df['diff_long'] < 5.0)]
    df = df[(df['diff_lat'] > .001) & (df['diff_long'] > .001)]
    
    # remove any pickups/dropoffs not within nyc bounds
    df = df[(df['pickup_longitude'] < -72) & (df['pickup_longitude'] > -75)]
    df = df[(df['pickup_latitude'] < 42) & (df['pickup_latitude'] > 39)]
    df = df[(df['dropoff_longitude'] < -72) & (df['dropoff_longitude'] > -75)]
    df = df[(df['dropoff_latitude'] < 42) & (df['dropoff_latitude'] > 39)]

    # remove invalid fare or passenger count
    df = df[(df['fare_amount'] > 2.50) & (df['fare_amount'] < 200) & (df['passenger_count'] <= 6) & (df['passenger_count'] > 0)] 
    return df
    
train_df = remove_outliers(train_df)
len(train_df)
plt.scatter(train_df[:10000]['manhattan_dist'], train_df[:10000]['fare_amount'])
plt.xlabel('manhattan distance')
plt.ylabel('fare')
plt.show()
train_df.describe()
def convert_to_one_hot (column, num_buckets, df, starting_index = 0):
    df_size = df.shape[0]
    one_hots = np.zeros((df_size, num_buckets), dtype='byte')
    one_hots[np.arange(df_size), df[column].values - starting_index] = 1
    return one_hots
year = convert_to_one_hot('year', 7, train_df, 2009)
hour = convert_to_one_hot('hour', 24, train_df, 0)
train_df.shape
def bucketize_feature(df,column):
    # split rides into 10 bins where 10% of rides were
    # use the quantile splits from train_df data
    buckets = train_df[column].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]).values
    bins = np.array(df[column].values)
    
    # set bin number
    lower_bound = -100000
    for i in range(buckets.shape[0]):
        upper_bound = buckets[i]
        bins[(bins >= lower_bound) & (bins < upper_bound)] = i
        lower_bound = upper_bound
    bins[(bins < 0) | (bins > 8)] = 9
    bins = np.array(bins, dtype='byte')

    return bins
p_long = bucketize_feature(train_df, 'pickup_longitude')
p_lat = bucketize_feature(train_df, 'pickup_latitude')
d_long = bucketize_feature(train_df, 'dropoff_longitude')
d_lat = bucketize_feature(train_df, 'dropoff_latitude')
print(p_long)
print(p_lat)
def feature_cross(a1, a2):
    rows = a1.shape[0]
    # 10 buckets for each, means 10*10 columns in feature cross
    cols = 100
    cross = np.zeros((rows, cols), dtype='byte')
    cross[np.arange(rows), (a1 * 10) + a2] = 1
    return cross

# cross latitudes and longitudes to get 1-hot vector representing grid of nyc
p_lat_x_long = feature_cross(p_lat, p_long)
d_lat_x_long = feature_cross(d_lat, d_long)
unique, counts = np.unique(p_long, return_counts=True)
print (np.asarray((unique, counts)).T)
unique, counts = np.unique(p_lat, return_counts=True)
print (np.asarray((unique, counts)).T)
print (p_lat_x_long.shape)
print (d_lat_x_long.shape)
print (year.shape)
print (hour.shape)
print (train_df['manhattan_dist'].shape)
# combine engineered features to create input layer
manhattan = train_df['manhattan_dist'].values.reshape(len(train_df), 1)
jfk_p = train_df['jfk_dist_pickup'].values.reshape(len(train_df), 1)
jfk_d = train_df['jfk_dist_dropoff'].values.reshape(len(train_df), 1)
lga_p = train_df['lga_dist_pickup'].values.reshape(len(train_df), 1)
lga_d = train_df['lga_dist_dropoff'].values.reshape(len(train_df), 1)
ewr_p = train_df['ewr_dist_pickup'].values.reshape(len(train_df), 1)
ewr_d = train_df['ewr_dist_dropoff'].values.reshape(len(train_df), 1)

train_X = np.concatenate((p_lat_x_long, d_lat_x_long, year, hour, manhattan, jfk_p, jfk_d, lga_p, lga_d, ewr_p, ewr_d), axis=1)
train_y = train_df['fare_amount'].values
print(train_X.shape)
print(train_y.shape)
validate_df = pd.read_csv('../input/train.csv', skiprows=range(1,10000001), nrows=10000, dtype=datatypes)
distance_between_points(validate_df)
validate_df = remove_outliers(validate_df)

def extract_features(df):
    #preprocess data, extract features we care about
    extract_date_details(df)
    p_lo = bucketize_feature(df, 'pickup_longitude')
    p_la = bucketize_feature(df, 'pickup_latitude')
    d_lo = bucketize_feature(df, 'dropoff_longitude')
    d_la = bucketize_feature(df, 'dropoff_latitude')
    p_la_x_lo = feature_cross(p_la, p_lo)
    d_la_x_lo = feature_cross(d_la, d_lo)
    yr = convert_to_one_hot('year', 7, df, 2009)
    hr = convert_to_one_hot('hour', 24, df, 0)
    manhattan = df['manhattan_dist'].values.reshape(len(df), 1)
    jfk_p = df['jfk_dist_pickup'].values.reshape(len(df), 1)
    jfk_d = df['jfk_dist_dropoff'].values.reshape(len(df), 1)
    lga_p = df['lga_dist_pickup'].values.reshape(len(df), 1)
    lga_d = df['lga_dist_dropoff'].values.reshape(len(df), 1)
    ewr_p = df['ewr_dist_pickup'].values.reshape(len(df), 1)
    ewr_d = df['ewr_dist_dropoff'].values.reshape(len(df), 1)

    print (p_la_x_lo.shape)
    print (d_la_x_lo.shape)
    print (yr.shape)
    print (hr.shape)
    print (manhattan.shape)

    X = np.concatenate((p_la_x_lo, d_la_x_lo, yr, hr, manhattan, jfk_p, jfk_d, lga_p, lga_d, ewr_p, ewr_d), axis=1)
    return X

X = extract_features(validate_df)
true_y = validate_df['fare_amount'].values
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Dense(128, activation='relu', input_dim=238))
# model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
# model.add(layers.BatchNormalization())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='adam',
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error
model.fit(train_X, train_y, epochs=15, batch_size=256,validation_split=0.1)
result = model.predict(X).flatten()
mean_y = np.mean(train_df['fare_amount'].values)
# result[result > 100] = mean_y
diff = true_y - result
mse = np.sum(diff ** 2) / len(diff)
rmse = np.sqrt(mse)
print (rmse)
distance_between_points(test_df)
X_test = extract_features(test_df)
pred_y_test = model.predict(X_test).flatten()
print (max(pred_y_test))
print (min(pred_y_test))
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission['fare_amount'] = pd.Series(pred_y_test)
sample_submission.to_csv('nn_submission.csv', index=False)

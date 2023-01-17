import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv', nrows=1000000)
train_df.describe()
test_df = pd.read_csv('../input/test.csv')
test_df.describe()
# use values that are most present. put remaining values in "outlier" buckets
long_range = -73.6 - (-74.05)
lat_range = 41 - 40.5
long_bucket_width = long_range / 25
lat_bucket_width = lat_range / 25
print (long_bucket_width)
print (lat_bucket_width)
def distance_between_points(df):
    df['diff_lat'] = abs(df['dropoff_latitude'] - df['pickup_latitude'])
    df['diff_long'] = abs(df['dropoff_longitude'] - df['pickup_longitude'])
    df['manhattan_dist'] = df['diff_lat'] + df['diff_long']
    
distance_between_points(train_df)
distance_between_points(test_df)
train_df['manhattan_dist'].values
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
    df = df[(df['pickup_longitude'] < -73.6) & (df['pickup_longitude'] > -74.05)]
    df = df[(df['pickup_latitude'] < 41) & (df['pickup_latitude'] > 40.5)]
    df = df[(df['dropoff_longitude'] < -73.6) & (df['dropoff_longitude'] > -74.05)]
    df = df[(df['dropoff_latitude'] < 41) & (df['dropoff_latitude'] > 40.5)]
#     df = df[(df['pickup_longitude'] < -72) & (df['pickup_longitude'] > -75)]
#     df = df[(df['pickup_latitude'] < 42) & (df['pickup_latitude'] > 39)]
#     df = df[(df['dropoff_longitude'] < -72) & (df['dropoff_longitude'] > -75)]
#     df = df[(df['dropoff_latitude'] < 42) & (df['dropoff_latitude'] > 39)]

    # remove invalid fare or passenger count
    df = df[(df['fare_amount'] > 2.50) & (df['passenger_count'] <= 6) & (df['passenger_count'] > 0)] 
    return df
    
train_df = remove_outliers(train_df)
len(train_df)
train_df.describe()
def get_mean_fare(col, val, df):
    filtered_df = df[df[col] == val]
    return np.mean(filtered_df['fare_amount'].values)

def graph_column_values (column_name, possible_values, df):
    average_fare_mapping = {}
    for val in possible_values:
        average_fare_mapping[val] = get_mean_fare(column_name, val, df)
    plt.bar(average_fare_mapping.keys(), average_fare_mapping.values())
    plt.xlabel(column_name)
    plt.ylabel('Average fare')
    plt.show()
graph_column_values('passenger_count', range(1, 7), train_df)
graph_column_values('year', range(2009, 2016), train_df)
graph_column_values('month', range(1, 13), train_df)
graph_column_values('day', range(0, 7), train_df)
graph_column_values('hour', range(0, 24), train_df)
def convert_to_one_hot (column, num_buckets, df, starting_index = 0):
    df_size = df.shape[0]
    one_hots = np.zeros((df_size, num_buckets), dtype='byte')
    one_hots[np.arange(df_size), df[column].values - starting_index] = 1
    print(one_hots)
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
p_lat_x_long[0]
unique, counts = np.unique(p_long, return_counts=True)
print (np.asarray((unique, counts)).T)
unique, counts = np.unique(p_lat, return_counts=True)
print (np.asarray((unique, counts)).T)
plt.scatter(train_df[:10000]['manhattan_dist'], train_df[:10000]['fare_amount'])
plt.xlabel('manhattan distance')
plt.ylabel('fare')
plt.show()
print (p_lat_x_long.shape)
print (d_lat_x_long.shape)
print (year.shape)
print (hour.shape)
print (train_df['manhattan_dist'].shape)
# linear model, no engineered features
# train_X = np.vstack((train_df['diff_lat'], train_df['diff_long'], np.ones(len(train_df)))).T
# train_y = train_df['fare_amount']
# print(train_X.shape)
# print(train_y.shape)

# linear model, engineered features
manhattan = train_df['manhattan_dist'].values.reshape(len(train_df), 1)
ones = np.ones((len(train_df), 1))
train_X = np.concatenate((p_lat_x_long, d_lat_x_long, year, hour, manhattan, ones), axis=1)
train_y = train_df['fare_amount']
print(train_X.shape)
print(train_y.shape)
(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond=None)
print(w)
np.dot(train_X[0], w)
validate_df = pd.read_csv('../input/train.csv', skiprows=range(1,1000001), nrows=10000)
mean_y = np.mean(train_df['fare_amount'].values)
mean_y
#validate_df = remove_outliers(validate_df)

def predict_price(df):
    #preprocess data, extract features we care about
    distance_between_points(df)
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
    ones = np.ones((len(df), 1))

    print (p_la_x_lo.shape)
    print (d_la_x_lo.shape)
    print (yr.shape)
    print (hr.shape)
    print (manhattan.shape)
    print (ones.shape)

    X = np.concatenate((p_la_x_lo, d_la_x_lo, yr, hr, manhattan, ones), axis=1)
    pred_y = np.dot(X, w)

    #replace outlier values
    pred_y[pred_y > 100] = mean_y
    print (pred_y.shape)
    return X, pred_y

X, pred_y = predict_price(validate_df)
true_y = validate_df['fare_amount']
#calc rmse
diff = true_y - pred_y
mse = np.sum(diff ** 2) / len(diff)
rmse = np.sqrt(mse)
print (rmse)
train_df[train_df['fare_amount'] > 100]
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(128, activation='relu'))
# Add another:
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='adam',
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error
model.fit(train_X, train_y.values, epochs=10, batch_size=100)
result = model.predict(X).flatten()
result[result > 100] = mean_y
diff = true_y - result
mse = np.sum(diff ** 2) / len(diff)
rmse = np.sqrt(mse)
print (rmse)
X, pred_test_y = predict_price(test_df)
nn_pred = model.predict(X)
nn_pred.shape
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission['fare_amount'] = pd.Series(pred_test_y)
sample_submission.to_csv('linear_submission.csv', index=False)
sample_submission['fare_amount'] = pd.Series(nn_pred)
sample_submission.to_csv('nn_submission.csv', index=False)

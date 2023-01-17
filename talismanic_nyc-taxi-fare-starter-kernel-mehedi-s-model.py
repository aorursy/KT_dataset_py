# Initial Python environment setup...
import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import os # reading the input files we have access to
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir('../input'))
# df = pd.DataFrame(index=pd.DatetimeIndex(start=dt.datetime(2016,1,1,0,0,1),
#     end=dt.datetime(2016,1,2,0,0,1), freq='H'))\
#     .reset_index().rename(columns={'index':'datetime'})

# df.head()
# df.dtypes
# df['ts'] = df.datetime.values.astype(np.int64) // 10 ** 9
# df.head()
# df.dtypes
# # t = pd.Timestamp(2017, 1, 1, 12)
# # ts = t.astye
train_df =  pd.read_csv('../input/train.csv', nrows = 10_000_000)
train_df.dtypes
#striping the timezone
def convert_to_datetime(df):
    test_time = df['pickup_datetime'].astype(str).str[:-4]
    df['date_time'] =  pd.to_datetime(test_time, format='%Y%m%d %H:%M:%S')
    return df
    
# converting the object to date time
train_df = convert_to_datetime(train_df)
# train_df.dtypes
train_df.head()
train_df.describe()
def normalize_fare_passenger(df):
    if 'fare_amount' in df.columns:
        print("old lenght: %d" %len(df))
        df = df[df.fare_amount>0]
    print("length after fare_amount normalization: %d" %len(df))
    df = df[df.passenger_count>0]
    print("length after passenger_count normalization: %d" %len(df))
    return df

train_df = normalize_fare_passenger(train_df)
train_df.head()
# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
    return df

train_df = add_travel_vector_features(train_df)
print(train_df.isnull().sum())
print('Old size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train_df))
train_df.head()
plot = train_df.iloc[1:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
print('Old size: %d' % len(train_df))
train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]
print('New size: %d' % len(train_df))
train_df[train_df.fare_amount<60].fare_amount.hist(bins=200, figsize=(14,3))
plt.xlabel('fare $USD')
plt.title('Histogram');
train_df.dtypes
#this is a kind of haversine formula to calculate the spherical distance
def haversine_distance_calculation(df):
    lon1, lat1, lon2, lat2 = df['pickup_longitude'], df['pickup_latitude'], df['dropoff_longitude'], df['dropoff_latitude']
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6367 * c
    df['distance'] = distance
    return df
train_df = haversine_distance_calculation(train_df)
train_df.head()
train_df.plot(x='distance', y='fare_amount', style = 'o')
def distance_fare_normalization(df):
    print("old lenght with distance greated than 50km: %d" %len(df))
    df = df[df.distance<50]
    print("length after distance normalization: %d" %len(df))
    if 'fare_amount' in df.columns:
        df = df[df.fare_amount<100]
        print("length after fare_amount normalization: %d" %len(df))
    return df

train_df = distance_fare_normalization(train_df)
train_df.head()
train_df.plot(x='distance', y='fare_amount', style = 'o')
def day_converter(df):
    day_of_year = df['date_time'].dt.dayofyear
    day_of_week = df['date_time'].dt.dayofweek
    df['day_of_year'] = day_of_year
    df['day_of_week'] = day_of_week
    return df

    
train_df = day_converter(train_df)
train_df.plot(x='day_of_year', y='fare_amount', style = 'o')
train_df.head()
# Construct and return an Nx3 input matrix for our linear model
# using the travel vector, plus a 1.0 for a constant bias term.
##MH:adding pickup_datetime in train_df
def get_input_matrix(df):
    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, df.distance, df.day_of_year, df.day_of_week,np.ones(len(df))))

train_X = get_input_matrix(train_df)
train_y = np.array(train_df['fare_amount'])

print(train_X.shape)
print(train_y.shape)
# The lstsq function returns several things, and we only care about the actual weight vector w.
(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond = None)
print(w)
w_OLS = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)), train_X.T), train_y)
print(w_OLS)
test_df = pd.read_csv('../input/test.csv')
test_df.dtypes
# Reuse the above helper functions to add our features and generate the input matrix.
add_travel_vector_features(test_df)
#
##converting to date time
test_df = convert_to_datetime(test_df)
##Normalizing fare_amount>0 & passenger_count>0
test_df = normalize_fare_passenger(test_df)
## dropping null values
test_df = test_df.dropna(how = 'any', axis = 'rows')
test_df =haversine_distance_calculation(test_df)
##normalization distance>50km and fare>100usd
test_df = distance_fare_normalization(test_df)
#converting date time to day vactor
test_df = day_converter(test_df)

test_X = get_input_matrix(test_df)
# Predict fare_amount on the test set using our model (w) trained on the training set.
test_y_predictions = np.matmul(test_X, w).round(decimals = 2)

# Write the predictions to a CSV file which we can submit to the competition.
submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': test_y_predictions},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)

print(os.listdir('.'))
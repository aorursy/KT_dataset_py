import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Read sample data 
df_train =  pd.read_csv('../input/train.csv', nrows = 100000, parse_dates=["pickup_datetime"])
df_train.head()
df_train.describe()
print('No of records b4 fare amount filter: ',len(df_train))
df_train = df_train[df_train.fare_amount >= 0]
print('No of records after fare amount filter ',  len(df_train))
print('No of records b4 passenger count filter ',  len(df_train))
df_train = df_train[df_train.passenger_count > 0]
print('No of records after passenger count filter ',  len(df_train))
print(df_train.isnull().sum())

#print('No of records b4 filter: ',len(df_train))
#df_train = df_train[df_train.pickup_longitude >= -180 & df_train.pickup_longitude <= 180]
#print('No of records after pickup longitude filter: ',  len(df_train))
#df_train = df_train[df_train.pickup_longitude >= -180 & df_train.pickup_longitude <= 180]
# minimum and maximum longitude test set
min(df_train.pickup_longitude.min(), df_train.dropoff_longitude.min()), \
max(df_train.pickup_longitude.max(), df_train.dropoff_longitude.max())
# minimum and maximum latitude test
min(df_train.pickup_latitude.min(), df_train.dropoff_latitude.min()), \
max(df_train.pickup_latitude.max(), df_train.dropoff_latitude.max())
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(df_train)
df_train.head()
print(df_train.isnull().sum())
df_train.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
df_train = df_train[(df_train.abs_diff_longitude < 5.0) & (df_train.abs_diff_latitude < 5.0)]

# Construct and return an Nx3 input matrix for our linear model
# using the travel vector, plus a 1.0 for a constant bias term.
def get_input_matrix(df):
    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df))))

train_X = get_input_matrix(df_train)
train_y = np.array(df_train['fare_amount'])

print(train_X.shape)
print(train_y.shape)
# The lstsq function returns several things, and we only care about the actual weight vector w.
(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond = None)
print(w)
df_test = pd.read_csv('../input/test.csv')
df_test.dtypes
# Reuse the above helper functions to add our features and generate the input matrix.
import os
add_travel_vector_features(df_test)
test_X = get_input_matrix(df_test)
# Predict fare_amount on the test set using our model (w) trained on the training set.
test_y_predictions = np.matmul(test_X, w).round(decimals = 2)

# Write the predictions to a CSV file which we can submit to the competition.
submission = pd.DataFrame(
    {'key': df_test.key, 'fare_amount': test_y_predictions},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)

print(os.listdir('.'))

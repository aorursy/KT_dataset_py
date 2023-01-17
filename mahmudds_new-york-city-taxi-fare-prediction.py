# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Initial Python environment setup...

# import numpy as np # linear algebra

# import pandas as pd # CSV file I/O (e.g. pd.read_csv)

# import os # reading the input files we have access to



# print(os.listdir('../input'))
# Read train data

taxi_train = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = 10_000_000)
taxi_train.columns.to_list()
taxi_train.dtypes
# Read sample submission

taxi_sample_sub = pd.read_csv('../input/new-york-city-taxi-fare-prediction/sample_submission.csv')

taxi_sample_sub.head()
import matplotlib.pyplot as plt

# Plot a histogram

taxi_train.fare_amount.hist(bins=30, alpha=0.5)

plt.show() 
# Given a dataframe, add two new features 'abs_diff_longitude' and

# 'abs_diff_latitude' reprensenting the "Manhattan vector" from

# the pickup location to the dropoff location.

def add_travel_vector_features(df):

    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()

    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()



add_travel_vector_features(taxi_train)
print(taxi_train.isnull().sum())
print('Old size: %d' % len(taxi_train))

taxi_train = taxi_train.dropna(how = 'any', axis = 'rows')

print('New size: %d' % len(taxi_train))
plot = taxi_train.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
print('Old size: %d' % len(taxi_train))

taxi_train = taxi_train[(taxi_train.abs_diff_longitude < 5.0) & (taxi_train.abs_diff_latitude < 5.0)]

print('New size: %d' % len(taxi_train))
# Construct and return an Nx3 input matrix for our linear model

# using the travel vector, plus a 1.0 for a constant bias term.

def get_input_matrix(df):

    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df))))



taxi_train_X = get_input_matrix(taxi_train)

taxi_train_y = np.array(taxi_train['fare_amount'])



print(taxi_train_X.shape)

print(taxi_train_y.shape)
# The lstsq function returns several things, and we only care about the actual weight vector w.

(w, _, _, _) = np.linalg.lstsq(taxi_train_X, taxi_train_y, rcond = None)

print(w)
w_OLS = np.matmul(np.matmul(np.linalg.inv(np.matmul(taxi_train_X.T, taxi_train_X)), taxi_train_X.T), taxi_train_y)

print(w_OLS)
# Read test data

taxi_test = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv')

taxi_test.columns.to_list()
# Reuse the above helper functions to add our features and generate the input matrix.

add_travel_vector_features(taxi_test)

taxi_test_X = get_input_matrix(taxi_test)

# Predict fare_amount on the test set using our model (w) trained on the training set.

taxi_test_y_predictions = np.matmul(taxi_test_X, w).round(decimals = 2)



# Write the predictions to a CSV file which we can submit to the competition.

submission = pd.DataFrame(

    {'key': taxi_test.key, 'fare_amount': taxi_test_y_predictions},

    columns = ['key', 'fare_amount'])

submission.to_csv('submission.csv', index = False)



print(os.listdir('.'))
# from sklearn.linear_model import LinearRegression

# # Create a LinearRegression object

# lr = LinearRegression()
# # Fit the model on the train data

# lr.fit(X=taxi_train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']],

# y=taxi_train['fare_amount']) 
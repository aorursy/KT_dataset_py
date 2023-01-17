import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
trn_data = pd.read_csv("../input/train.csv", nrows = 2_000_000, parse_dates=["pickup_datetime"])

tst_data = pd.read_csv("../input/test.csv")
trn_data.head()
print(trn_data.shape)
print(trn_data.isnull().sum())
trn_data = trn_data.dropna(how = 'any', axis = 'rows')

trn_data.isnull().values.any()
tst_data.head()
tst_data.shape
sample_sub = pd.read_csv("../input/sample_submission.csv")

sample_sub.head()
from sklearn.linear_model import LinearRegression



# Create a linear regression object

lr = LinearRegression()



# Fit the model on the train data

lr.fit(X = trn_data[['pickup_longitude', 'pickup_latitude', 

                     'dropoff_longitude', 'dropoff_latitude', 'passenger_count']],

       y = trn_data['fare_amount'])
# Select variables with which the model has been trained

features = ['pickup_longitude', 'pickup_latitude', 

            'dropoff_longitude', 'dropoff_latitude', 'passenger_count']



tst_data['fare_amount'] = lr.predict(tst_data[features])
my_submission = tst_data[['key', 'fare_amount']]
my_submission.head()
my_submission.to_csv('final_submission_lr', index = False)

print(os.listdir('.'))
trn_data.describe()
trn_data = trn_data.drop(trn_data[(trn_data.fare_amount <= 2.5)].index, axis = 0)

trn_data = trn_data.drop(trn_data[(trn_data.passenger_count > 8)].index, axis = 0)

trn_data = trn_data.drop(trn_data[(trn_data.pickup_latitude < 40) | 

                          (trn_data.pickup_latitude > 42)].index, axis = 0)

trn_data = trn_data.drop(trn_data[(trn_data.pickup_longitude < -75) | 

                          (trn_data.pickup_longitude > -73)].index, axis = 0)

trn_data = trn_data.drop(trn_data[(trn_data.dropoff_latitude < 40) | 

                          (trn_data.dropoff_latitude > 42)].index, axis = 0)

trn_data = trn_data.drop(trn_data[(trn_data.dropoff_longitude < -75) | 

                          (trn_data.dropoff_longitude > -73)].index, axis = 0)

trn_data.describe()
def distance_between_points(df):

    df['diff_lat'] = abs(df['dropoff_latitude'] - df['pickup_latitude'])

    df['diff_long'] = abs(df['dropoff_longitude'] - df['pickup_longitude'])

    df['manhattan_dist'] = df['diff_lat'] + df['diff_long']

    

distance_between_points(trn_data)

distance_between_points(tst_data)
trn_data.describe()
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV



ridge = Ridge()



parameters = {'alpha':np.linspace(0,1,20)}



ridge_regressor = GridSearchCV(ridge, parameters, scoring ='neg_mean_squared_error', cv = 5)



ridge_regressor.fit(X = trn_data[['pickup_longitude', 'pickup_latitude', 

                     'dropoff_longitude', 'dropoff_latitude', 'passenger_count',

                     'manhattan_dist']],

       y = trn_data['fare_amount'])



print(ridge_regressor.best_params_)

print(ridge_regressor.best_score_)
features = ['pickup_longitude', 'pickup_latitude', 

            'dropoff_longitude', 'dropoff_latitude', 'passenger_count',

           'manhattan_dist']



tst_data['fare_amount'] = ridge_regressor.predict(tst_data[features])



my_submission = tst_data[['key', 'fare_amount']]



my_submission.to_csv('final_submission_L2r', index = False)

print(os.listdir('.'))
from keras.models import Sequential

from keras.layers import Dense



# define base model

def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(10, input_dim = 6, kernel_initializer = 'normal', activation = 'sigmoid'))

    model.add(Dense(1, kernel_initializer = 'normal'))

    # Compile model

    model.compile(loss = 'mean_squared_error', optimizer = 'adam')

    return model
X = trn_data[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'manhattan_dist']]

y = trn_data['fare_amount']



model = baseline_model()

model.fit(X, y, epochs = 5, batch_size = 100, verbose = 1)



features = ['pickup_longitude', 'pickup_latitude', 

            'dropoff_longitude', 'dropoff_latitude', 'passenger_count',

           'manhattan_dist']



tst_data['fare_amount'] = model.predict(tst_data[features])



my_submission = tst_data[['key', 'fare_amount']]

my_submission.to_csv('final_submission_NN', index = False)

print(os.listdir('.'))
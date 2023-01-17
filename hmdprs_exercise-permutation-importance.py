# load data

import pandas as pd

data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=50000)



# remove data with extreme outlier coordinates or negative fares

data = data.query(

    'pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +

    'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +

    'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +

    'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +

    'fare_amount > 0'

)

y = data['fare_amount'].copy()



base_features = [

    'pickup_longitude',

    'pickup_latitude',

    'dropoff_longitude',

    'dropoff_latitude',

    'passenger_count'

]

X = data[base_features].copy()



# split train and valid data

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# define and fit model

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

first_model = RandomForestRegressor(n_estimators=50, random_state=1).fit(train_X, train_y)



# setup feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.ml_explainability.ex2 import *

print("Setup is completed.")



# show data

print("Data sample:")

data.head()
train_X.describe()
train_y.describe()
# check your answer (run this code cell to receive credit!)

q_1.solution()
# calculate importances

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)



# check your answer

q_2.check()



# visualize your results

from eli5 import show_weights

show_weights(perm, feature_names=base_features)
# uncomment the lines below for a hint or to see the solution

# q_2.hint()

# q_2.solution()
# check your answer (run this code cell to receive credit!)

q_3.solution()
# create new features

data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)

data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)



features_2  = [

    'pickup_longitude',

    'pickup_latitude',

    'dropoff_longitude',

    'dropoff_latitude',

    'abs_lat_change',

    'abs_lon_change'

]

X = data[features_2].copy()



# split train and valid data

new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)



# define and fit model

second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)



# create a PermutationImportance object on second_model and fit it to new_val_X and new_val_y

# use a random_state of 1 for reproducible results that match the expected solution

perm2 = PermutationImportance(second_model, random_state=1).fit(new_val_X, new_val_y)



# visualize your results

show_weights(perm2, feature_names = features_2)



# check your answer

q_4.check()
show_weights(perm2, feature_names = features_2)
# check your answer (run this code cell to receive credit!)

q_4.solution()
# check your answer (run this code cell to receive credit!)

q_5.solution()
# check your answer (run this code cell to receive credit!)

q_6.solution()
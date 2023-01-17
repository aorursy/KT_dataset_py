# Loading data, dividing, modeling and EDA below

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=50000)



# Remove data with extreme outlier coordinates or negative fares

data = data.query('pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +

                  'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +

                  'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +

                  'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +

                  'fare_amount > 0'

                  )



y = data.fare_amount



base_features = ['pickup_longitude',

                 'pickup_latitude',

                 'dropoff_longitude',

                 'dropoff_latitude',

                 'passenger_count']



X = data[base_features]





X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

first_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(X_train, y_train)



# Environment Set-Up for feedback system.

from learntools.core import binder

binder.bind(globals())

from learntools.ml_explainability.ex2 import *

print("Setup Complete")



# show data

print("Data sample:")

data.head()
X_train.describe()
y_train.describe()
#q_1.solution()
import eli5

from eli5.sklearn import PermutationImportance



# Make a small change to the code below to use in this problem. 

perm = PermutationImportance(first_model, random_state=1)

perm.fit(X_test, y_test)



q_2.check()



# uncomment the following line to visualize your results

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
# q_2.hint()

# q_2.solution()
#q_3.solution()
# create new features

data['abs_lon_change'] = abs(

    data.dropoff_longitude - data.pickup_longitude

)



data['abs_lat_change'] = abs(

    data.dropoff_latitude - data.pickup_latitude

)



features_2  = ['pickup_longitude',

               'pickup_latitude',

               'dropoff_longitude',

               'dropoff_latitude',

               'abs_lat_change',

               'abs_lon_change']



X = data[features_2]

new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(X, y, random_state=1)



second_model = RandomForestRegressor(n_estimators=30, random_state=1)

second_model.fit(new_X_train, new_y_train)



# Create a PermutationImportance object on second_model and fit it to new_X_test and new_y_test

# Use a random_state of 1 for reproducible results that match the expected solution.

perm2 = PermutationImportance(second_model, random_state = 1)

perm2.fit(new_X_test, new_y_test)



# show the weights for the permutation importance you just calculated

eli5.show_weights(perm2, feature_names = new_X_test.columns.tolist())



q_4.check()
#q_4.solution()
#q_5.solution()
#q_6.solution()
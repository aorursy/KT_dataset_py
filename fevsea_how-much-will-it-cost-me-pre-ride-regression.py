import pandas as pd

import numpy as np

import datetime as dt

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Parses the datetime string into datetime objects

def dateparse(x):

    try:

        return dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    except:

        return pd.NaT



# Load cleaned data

df = pd.read_csv("../input/chicago_taxi_trips_2016_05.csv", parse_dates=['trip_start_timestamp', 'trip_end_timestamp'],  date_parser=dateparse)

print(df.shape)
df.head()
# Fields not needed to our problem

to_drop = ["taxi_id",

           "pickup_census_tract",

           "dropoff_census_tract",

           "tips",

           "trip_seconds",

           "trip_miles",

           "extras",

           "trip_total",

           "company",

           "tolls",

           "payment_type",

           "trip_end_timestamp"]



# Drop selected fields in place

df.drop(to_drop, inplace=True, axis=1)
# For each feature I'm going to use for training, let's calculate and print

#the number and porcentage of missing values

features = ["trip_start_timestamp", "pickup_community_area", "dropoff_community_area", "fare", "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]

for f in features:

    na = df[f].isnull().sum()

    print(f, "->", "Missing values:", na, "Percentage:", na/len(df)*100)
df.dropna(inplace=True)

df.shape
# Transform the start datatime object into discrete weekday and time features

df['weekday'] = df['trip_start_timestamp'].map(lambda x: x.weekday())

df['time'] = df['trip_start_timestamp'].map(lambda x: x.hour*4 + round(x.minute/15))

df.drop('trip_start_timestamp', inplace=True, axis=1)
import geopy.distance



# Load lookup table

lt = pd.read_json("../input/column_remapping.json")



# Change indices with the real value

df['pickup_latitude'] = df['pickup_latitude'].map(lambda x: lt.pickup_latitude[x])

df['pickup_longitude'] = df['pickup_longitude'].map(lambda x: lt.pickup_longitude[x])

df['dropoff_latitude'] = df['dropoff_latitude'].map(lambda x: lt.dropoff_latitude[x])

df['dropoff_longitude'] = df['dropoff_longitude'].map(lambda x: lt.dropoff_longitude[x])
# Calculate lineal distance using coordinates

def calculate_distance(src):

    coords_1 = (src["pickup_latitude"], src["pickup_longitude"])

    coords_2 = (src["dropoff_latitude"],src["dropoff_longitude"])

    return geopy.distance.distance(coords_1, coords_2).m



# Generate lineal distance field

df['distance'] = df.apply(calculate_distance, axis='columns')
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.distplot(df["distance"]);
# Remove detected outliers

df = df[df.distance != 0]

df = df[df.distance <= 35000]

sns.distplot(df["distance"]);

df["distance"].describe()
# Since I've scaled the time feture to blocks of 15 minutes if divided by 4 obtain hours

sns.distplot(df["time"]/4);

(df["time"]/4).describe()
# Maybe some days have more trips than others

sns.countplot(df["weekday"]);

df["weekday"].describe()
# Number of trips for time for each weekday

sns.violinplot("weekday", "time", data=df)
# Plot dropoff latitude and longitude as map coordinates

sns.jointplot(y="dropoff_latitude", x="dropoff_longitude", data=df);
# Remove detected outliers and plot again

df = df[df.dropoff_longitude >= -87.85]

sns.jointplot(y="dropoff_latitude", x="dropoff_longitude", data=df);
# Plot pickup latitude and longitude as map coordinates

sns.jointplot(y="pickup_latitude", x="pickup_longitude", data=df);
# Remove detected outliers and plot again

df = df[df.pickup_longitude >= -87.85]

sns.jointplot(y="pickup_latitude", x="pickup_longitude", data=df)
# Generate fare histogram

sns.distplot(df["fare"]);

df["fare"].describe()
# Remove heavy outliers and plot again

df = df[df["fare"]<=55]

sns.distplot(df["fare"]);
# Plot histogram for pickup community area

sns.distplot(df["pickup_community_area"])

df["pickup_community_area"].describe()
# Plot dropoff community area to detect potential outliers

sns.distplot(df["dropoff_community_area"])

df["dropoff_community_area"].describe()
from sklearn.model_selection import train_test_split



X = df.drop('fare', axis=1)

y = df["fare"]

y = np.asarray(y, dtype=np.float64)



# Generate sets: 80% train, 20% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)
from sklearn.metrics import mean_squared_error

from sklearn.dummy import DummyRegressor

from sklearn.model_selection import GridSearchCV

import seaborn as sns



# Given a model calculate the train and test scores

def get_scores(est):

    y_train_predict = est.predict(X_train)

    train_predict = mean_squared_error(y_train, y_train_predict)

    y_test_predict = est.predict(X_test)

    test_predict = mean_squared_error(y_test, y_test_predict)

    print("Train mse:", train_predict, "; Test mse:", test_predict)

    return (train_predict, test_predict)



# Given a model plot the residual plot for the test dataset

def plot_residuals(est):

    sns.residplot(est.predict(X_test), y_test)



# Dummy model generation and score

dummyModel = DummyRegressor(strategy="mean")

dummyModel.fit(X_train, y_train)

get_scores(dummyModel)
from sklearn import tree



# Parameters to tune

parameters = {'min_samples_split':[64, 128, 256],

              'min_samples_leaf':[2, 4, 16]}



# Declare objects

est_dt_r = tree.DecisionTreeRegressor()

est_dt = GridSearchCV(est_dt_r, parameters, n_jobs=-1, cv=4, verbose=1)

# Train and score model

est_dt = est_dt.fit(X_train, y_train)

get_scores(est_dt)

plot_residuals(est_dt)
# Print best parameters found by the grid search

est_dt.best_params_
from sklearn.linear_model import SGDRegressor



# Parameters to tune

parameters = {'alpha':[0.001, 0.0001, 0.00001]}



# Declare objects

est_nn_r = SGDRegressor(max_iter=10000, tol=1e-3)

est_nn = GridSearchCV(est_nn_r, parameters, n_jobs=-1, cv=4, verbose=1)

# Train and score model

est_nn = est_nn.fit(X_train, y_train)

get_scores(est_nn)

plot_residuals(est_nn)
from sklearn.ensemble import RandomForestRegressor



# Parameters to tune

parameters = {'n_estimators':[10, 50, 100, 150],

              'min_samples_split':[64, 128, 256],

              'min_samples_leaf': [2, 4, 6]}



# Declare objects

est_rf_r = RandomForestRegressor(random_state=1337)

est_rf = GridSearchCV(est_rf_r, parameters, cv=4, verbose=1, n_jobs=-1)

# Train and score model

est_rf = est_rf.fit(X_train, y_train)

get_scores(est_rf)

plot_residuals(est_rf)
 # Print best parameters found by the grid search

est_rf.best_params_
from sklearn.ensemble import AdaBoostRegressor



# Parameters to tune

parameters = {'n_estimators':[25, 50, 100],

              'loss':["square", "linear"],

              'learning_rate': [0.75, 1, 1.25]}



# Declare objects

est_ada_r = AdaBoostRegressor(random_state=1337)

est_ada = GridSearchCV(est_ada_r, parameters, cv=4, verbose=1, n_jobs=-1)

# Train and score model

est_ada = est_ada.fit(X_train, y_train)

get_scores(est_ada)

plot_residuals(est_ada)
# Print best parameters found by the grid search

est_ada.best_params_
from sklearn.ensemble import GradientBoostingRegressor



# Parameters to tune

parameters = {'n_estimators':[50, 100, 150],

              'min_samples_split':[64, 128, 256],

              'min_samples_leaf': [2, 4, 6]}



# Declare objects

est_gtb_r = GradientBoostingRegressor(loss='ls')

est_gtb = GridSearchCV(est_gtb_r, parameters, cv=4, verbose=1, n_jobs=-1)

# Train and score model

est_gtb = est_gtb.fit(X_train, y_train)

get_scores(est_gtb)

plot_residuals(est_gtb)

# Print best parameters found by the grid search

print(est_gtb.best_params_)
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler



# Fit and transform the scaler with the train data

scaler = StandardScaler()

scaler.fit(X_train)

X_train_s = scaler.transform(X_train)

X_test_s = scaler.transform(X_test)



# Define and train the model

est_net = MLPRegressor(learning_rate="adaptive", max_iter=400, hidden_layer_sizes=(150, ))

est_net = est_net.fit(X_train_s, y_train)
# Scale the test data and generate benchmark

y_train_predict = est_net.predict(X_train_s)

train_predict = mean_squared_error(y_train, y_train_predict)

y_test_predict = est_net.predict(X_test_s)

test_predict = mean_squared_error(y_test, y_test_predict)

print("Train mse:", train_predict, "; Validation mse:", test_predict)



sns.residplot(est_net.predict(X_test_s), y_test);
from sklearn.ensemble import RandomForestRegressor

est = RandomForestRegressor(random_state=1337, min_samples_split=60, n_estimators=300, n_jobs=-1)              

est = est.fit(X_train, y_train)

get_scores(est)

plot_residuals(est)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_test_predict = est.predict(X_test)



mae = mean_absolute_error(y_test, y_test_predict)

mse = mean_squared_error(y_test, y_test_predict)

r_2 = r2_score(y_test, y_test_predict)

print("MSE -> ", mse)

print("MAE -> ", mae)

print("R^2 -> ", r_2)
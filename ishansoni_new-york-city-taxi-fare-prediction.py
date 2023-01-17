# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Basic EDA libraries
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('dark_background')
sns.set_style("whitegrid")
from IPython.display import display
%matplotlib inline

# Basic ML Libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

import xgboost as xgb

# Deep Learning Libraries

# some other libraries
import geopy.distance
from geopy.geocoders import Nominatim
# Since the train data is pretty large, we will import only a random subset of rows to do our analysis.

# The data to load
train_file = "../input/train.csv"

# Take every N-th (in this case 10th) row
n = 10

# Count the lines or use an upper bound
num_lines = sum(1 for l in open(train_file))

# The row indices to skip - make sure 0 is not included to keep the header!
skip_idx = [x for x in range(1, num_lines) if x % n != 0]

# Read the data
# train = pd.read_csv(train_file, dtype={'fare_amount': 'float32', 'pickup_longitude' : 'float32', 'pickup_longitude' : 'float32', 'dropoff_longitude' : 'float32', 'dropoff_latitude' : 'float32', 'passenger_count' : 'int32'},
#                    skiprows = skip_idx, parse_dates = ['pickup_datetime']).drop(columns = 'key')

train = pd.read_csv(train_file, dtype={'fare_amount': 'float32', 'pickup_longitude' : 'float32', 'pickup_longitude' : 'float32', 'dropoff_longitude' : 'float32', 'dropoff_latitude' : 'float32', 'passenger_count' : 'int32'},
                    nrows = 2_000_000, parse_dates = ['pickup_datetime']).drop(columns = 'key')


test = pd.read_csv("../input/test.csv", dtype={'fare_amount': 'float32', 'pickup_longitude' : 'float32', 'pickup_longitude' : 'float32', 'dropoff_longitude' : 'float32', 'dropoff_latitude' : 'float32', 'passenger_count' : 'int32'},
                   parse_dates = ['pickup_datetime'])

# To be used for creating the subission csv
test_id = list(test.pop('key'))

display(train.sample(n = 5))

display(test.sample(n = 5))

display(train.info())

display(test.info())
# Original Shape
print(train.shape)
print(test.shape)
# Lets have a peek at our data's descriptive statistics 
train.describe()
# Check for nulls in our dataset.
print("Training data has nulls? :", train.isnull().values.any())
print("Testing data has nulls? :", test.isnull().values.any())
print(train.isnull().sum())
# Since there aren't a lot of rows with nulls, we will drop them all
train = train.dropna(how = 'any', axis = 'rows')
# Let's have a look at the fare's distribution

sns.distplot(train.sample(n = 20000)["fare_amount"], hist = True, kde = True)
fig = plt.gcf()
fig.set_size_inches(20, 8)
plt.show()
# Most of the fare amount lies b/w 0 and 60. Lets remove some extreme outliers
# Base Fare for NYC Cab : 2.5$ (http://nymag.com/nymetro/urban/features/taxi/n_20286/)

q75, q25 = np.percentile(train["fare_amount"], [75 ,25])
iqr = q75 - q25

print("Fare Iqr", iqr)

train = train[train["fare_amount"].between(left = 2.5, right = (q75 + 10 * iqr))]
# Cleaning Latitudes & Longitudes
# Latitudes range from -90 to 90.
# Longitudes range from -180 to 180.
# New york lat/long =>40.730610, -73.935242

# Lets have a look at all Invalid lat/long ranges

display(train[np.logical_or(train["pickup_latitude"] < -90, train["pickup_latitude"] > 90)])
display(train[np.logical_or(train["dropoff_latitude"] < -90, train["dropoff_latitude"] > 90)])

display(train[np.logical_or(train["pickup_longitude"] < -180, train["pickup_longitude"] > 180)])
display(train[np.logical_or(train["dropoff_longitude"] < -180, train["dropoff_longitude"] > 180)])
#There are a few rows that have invalid Latitudes & Longitudes. Will will discard them. 
#Plus, we will also discard rows that have lat/long ranges not possible for the NY region (40.730610, -73.935242)

train = train[np.logical_and(train["pickup_latitude"] >= 40, train["pickup_latitude"] <= 42)]
train = train[np.logical_and(train["dropoff_latitude"] >= 40, train["dropoff_latitude"] <= 42)]
train = train[np.logical_and(train["pickup_longitude"] >= -75, train["pickup_longitude"] <= -73)]
train = train[np.logical_and(train["dropoff_longitude"] >= -75, train["dropoff_longitude"] <= -73)]
# Lets have a look at the Passenger Count distribution

sns.distplot(train.sample(n = 20000)["passenger_count"], hist = True, kde = True)
fig = plt.gcf()
fig.set_size_inches(20, 8)
plt.show()
# Most of the passenger counts are between 0 and 6. We will remove all others.

train = train[train["passenger_count"].between(left = 0, right = 6)]
print(train.shape)
print(test.shape)
# Example
coords_1 = (52.2296756, 21.0122287)
coords_2 = (52.406374, 16.9251681)

print(geopy.distance.vincenty(coords_1, coords_2).km)
def distanceCalculator(row) :
    c1 = (row["pickup_latitude"], row["pickup_longitude"])
    c2 = (row["dropoff_latitude"], row["dropoff_longitude"])
    
    return geopy.distance.vincenty(c1, c2).km

train["distance"] = train.apply(distanceCalculator, axis = 1)
test["distance"] = test.apply(distanceCalculator, axis = 1)
train.sample(n = 5)
train["hour"] = train["pickup_datetime"].dt.hour
test["hour"] = test["pickup_datetime"].dt.hour

train["dayOfWeek"] = train["pickup_datetime"].dt.dayofweek
test["dayOfWeek"] = test["pickup_datetime"].dt.dayofweek

train['day'] = train['pickup_datetime'].dt.day
test['day'] = test['pickup_datetime'].dt.day

train['month'] = train['pickup_datetime'].dt.month
test['month'] = test['pickup_datetime'].dt.month

train["year"] = train["pickup_datetime"].dt.year
test["year"] = test["pickup_datetime"].dt.year
train.sample(n = 5)
def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):  
    distance = np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)
    
    return distance

def transform(data):
    # Distances to nearby airports, and city center
    # By reporting distances to these points, the model can somewhat triangulate other locations of interest
    nyc = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)
   
    data['distance_to_center'] = dist(nyc[1], nyc[0],
                                      data['pickup_latitude'], data['pickup_longitude'])
    data['pickup_distance_to_jfk'] = dist(jfk[1], jfk[0],
                                         data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_jfk'] = dist(jfk[1], jfk[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_ewr'] = dist(ewr[1], ewr[0], 
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_ewr'] = dist(ewr[1], ewr[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_lgr'] = dist(lgr[1], lgr[0],
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_lgr'] = dist(lgr[1], lgr[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    
    data['long_dist'] = data['pickup_longitude'] - data['dropoff_longitude']
    data['lat_dist'] = data['pickup_latitude'] - data['dropoff_latitude']
    
    return data


train = transform(train)
test = transform(test)
train.sample(n = 5)
f, ax = plt.subplots(1, 2, figsize = (20, 8))

sns.countplot(x = "hour", data = train.sample(n = 20000), ax = ax[0])
plt.xlabel("Hour of the day")
plt.ylabel("Cab frequency")

sns.barplot(x = "hour", y = "fare_amount", data = train.sample(n = 20000), ax = ax[1])
plt.xlabel("Hour of the day")
plt.ylabel("Average Fare amount")
plt.show()
f, ax = plt.subplots(1, 2, figsize = (20, 8))

sns.countplot(x = "dayOfWeek", data = train.sample(n = 20000), ax = ax[0])
plt.xlabel("Day of the week")
plt.ylabel("Cab frequency")

sns.barplot(x = "dayOfWeek", y = "fare_amount", data = train.sample(n = 20000), ax = ax[1])
plt.xlabel("Day of the week")
plt.ylabel("Average Fare amount")
plt.show()
sns.factorplot(x = "hour", y = "fare_amount", hue = "dayOfWeek", data = train.sample(n = 20000))
fig = plt.gcf()
fig.set_size_inches(20, 12)
plt.show()
train["year"].value_counts()
# The fare data contains data points for several years, The fares shoule depend on it(Inflation!)

sns.barplot(x = "year", y = "fare_amount", data = train.sample(n = 20000))
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.show()
# Passenger Count
sns.barplot(x = "passenger_count", y = "fare_amount", data = train.sample(n = 20000))
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.show()
display(train.sample(n = 5))
display(test.sample(n = 5))
target = train["fare_amount"].values
train = train.drop(columns = ["fare_amount", "pickup_datetime"], axis = 1)
test = test.drop(columns = ["pickup_datetime"], axis = 1)
display(train.sample(n = 2))
display(test.sample(n = 2))
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.15)
# Even though I have created the pipeline to optimise hyperparameters and save a list of model, I will not be using it
# due to the sheer data size. 

modelResults = pd.DataFrame(columns = ['Model_Name', 'Model', 'Params', 'Test_Score', 'CV_Mean', 'CV_STD'])

def save(grid, modelName, calFI):
    global modelResults
    cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv = 10, scoring = 'neg_mean_squared_error')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    test_score = grid.score(X_test, y_test)
    
    print("Best model parameter are\n", grid.best_estimator_)
    print("Saving model {}\n".format(modelName))
    print("Mean Cross validation score is {} with a Standard deviation of {}\n".format(cv_mean, cv_std))
    print("Test Score for the model is {}\n".format(test_score))
    
    if calFI:
        pd.Series(grid.best_estimator_.feature_importances_, train.columns).sort_values(ascending = True).plot.barh(width = 0.6)
        fig = plt.gcf()
        fig.set_size_inches(12, 12)
        plt.title("{} Feature Importance".format(modelName))
        plt.show()
    
    
    modelResults = modelResults.append({'Model_Name' : modelName, 'Model' : grid.best_estimator_, 'Params' : grid.best_params_, 'Test_Score' : test_score, 'CV_Mean' : cv_mean, 'CV_STD' : cv_std}
                                       , ignore_index=True)
    
    
def doGridSearch(classifier, params):
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    score_fn = make_scorer(mean_squared_error)
    grid = GridSearchCV(classifier, params, scoring = score_fn, cv = cv)
    grid = grid.fit(X_train, y_train)
    
    return grid    
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

random_forest = RandomForestRegressor(max_features = None, oob_score = True, 
                                      bootstrap = True, verbose = 1, n_jobs = -1)

random_forest.fit(X_train, y_train)
y_test_preds = random_forest.predict(X_test)

print("RMSE score :", np.sqrt(mean_squared_error(y_test, y_test_preds)))
random_forest_preditions = random_forest.predict(test)
sub_rf = pd.DataFrame({'key': test_id, 'fare_amount': random_forest_preditions})
sub_rf.to_csv('rf_nyc.csv', index = False)
#Cross-validation

params = {
    # Parameters that we are going to tune.
    'max_depth': 8, #Result of tuning with CV
    'eta':.03, #Result of tuning with CV
    'subsample': 0.8, #Result of tuning with CV
    'colsample_bytree': 0.8, #Result of tuning with CV
    # Other parameters
    'objective':'reg:linear',
    'eval_metric':'rmse',
    'silent': 1
}

#Block of code used for hypertuning parameters. Adapt to each round of parameter tuning.
#Turn off CV in submission

CV = False
if CV:
    dtrain = xgb.DMatrix(train,label=y)
    gridsearch_params = [
        (eta)
        for eta in np.arange(.04, 0.12, .02)
    ]

    # Define initial best params and RMSE
    min_rmse = float("Inf")
    best_params = None
    for (eta) in gridsearch_params:
        print("CV with eta={} ".format(
                                 eta))

        # Update our parameters
        params['eta'] = eta

        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=3,
            metrics={'rmse'},
            early_stopping_rounds=10
        )

        # Update best RMSE
        mean_rmse = cv_results['test-rmse-mean'].min()
        boost_rounds = cv_results['test-rmse-mean'].argmin()
        print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = (eta)

    print("Best params: {}, RMSE: {}".format(best_params, min_rmse))
else:
    #Print final params to use for the model
    params['silent'] = 0 #Turn on output
    print(params)
def XGBmodel(x_train,x_test,y_train,y_test,params):
    matrix_train = xgb.DMatrix(x_train, label = y_train)
    matrix_test = xgb.DMatrix(x_test, label = y_test)
    model = xgb.train(params = params,
                    dtrain = matrix_train,num_boost_round = 5000, 
                    early_stopping_rounds = 10,evals = [(matrix_test,'test')])
    return model

model = XGBmodel(X_train, X_test, y_train, y_test, params)
xgbpredictions = model.predict(xgb.DMatrix(test), ntree_limit = model.best_ntree_limit)
sub_xgb = pd.DataFrame({'key': test_id, 'fare_amount': xgbpredictions})
sub_xgb.to_csv('xgboost_nyc.csv', index = False)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



train = pd.read_csv("../input/train.csv")

testfile = pd.read_csv("../input/test.csv")

train.head(6)
train.info()

correlations_data = train.corr()['trip_duration'].sort_values()

correlations_data
train.describe()
plt.subplots(figsize=(18,6))

plt.title("Visualisation des outliers")

train.boxplot();
#We only keep rows with a trip_duration between 100 and 10000 seconds.

train = train[(train.trip_duration < 10000) & (train.trip_duration > 100)]
#Removing 'id' and store_and_fwd_flag' columns

train.drop(['id'], axis=1, inplace=True)

train.drop(['store_and_fwd_flag'], axis=1, inplace=True)

testfile.drop(['store_and_fwd_flag'], axis=1, inplace=True)
#Datetyping the dates so we can work with it

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)

testfile['pickup_datetime'] = pd.to_datetime(testfile.pickup_datetime)

train.info()
#Date features creations and deletions

train['week'] = train.pickup_datetime.dt.week

train['weekday'] = train.pickup_datetime.dt.weekday

train['hour'] = train.pickup_datetime.dt.hour

train.drop(['pickup_datetime'], axis=1, inplace=True)

train.drop(['dropoff_datetime'], axis=1, inplace=True)

testfile['week'] = testfile.pickup_datetime.dt.week

testfile['weekday'] = testfile.pickup_datetime.dt.weekday

testfile['hour'] = testfile.pickup_datetime.dt.hour

testfile.drop(['pickup_datetime'], axis=1, inplace=True)
#Visualising the distribution of trip_duration values

plt.subplots(figsize=(18,6))

plt.hist(train['trip_duration'].values, bins=100)

plt.xlabel('trip_duration')

plt.ylabel('number of train records')

plt.show()
#Log transformation

plt.subplots(figsize=(18,6))

train['trip_duration'] = np.log(train['trip_duration'].values) #+1 is not needed here as our trip_duration values are all positive and not normalized. But it would be necessary to normalize and add 1 to make a robust code for new data.

plt.hist(train['trip_duration'].values, bins=100)

plt.xlabel('log(trip_duration)')

plt.ylabel('number of train records')

plt.show()
y = train["trip_duration"]

train.drop(["trip_duration"], axis=1, inplace=True)

X = train

X.shape, y.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
from sklearn.ensemble import RandomForestRegressor



#The randomforestregressor params are chosen in the following hyperparameters tuning

m1 = RandomForestRegressor(n_estimators=19, min_samples_split=2, min_samples_leaf=4, max_features='auto', max_depth=80, bootstrap=True)

m1.fit(X_train, y_train)

m1.score(X_valid, y_valid)
#from sklearn.ensemble import GradientBoostingRegressor



#gradient_boosted = GradientBoostingRegressor()

#gradient_boosted.fit(X_train, y_train)

#gradient_boosted.score(X_valid, y_valid)



#score: around 0.5
from sklearn.metrics import mean_squared_error as MSE



print(np.sqrt(MSE(y_valid, m1.predict(X_valid))))

#print(np.sqrt(MSE(y_valid, gradient_boosted.predict(X_valid))))

#from sklearn.model_selection import RandomizedSearchCV



#n_estimators = [int(x) for x in np.linspace(start = 5, stop = 20, num = 16)]

#max_features = ['auto', 'sqrt']

#max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

#max_depth.append(None)

#min_samples_split = [2, 5, 10]

#min_samples_leaf = [1, 2, 4]

#bootstrap = [True, False]



#random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}



#random_cv = RandomizedSearchCV(estimator = m1, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

#print(random_cv.best_params_)
test_columns = X_train.columns

predictions = m1.predict(testfile[test_columns])
my_submission = pd.DataFrame({'id': testfile.id, 'trip_duration': np.exp(predictions)})

my_submission.head()
my_submission.to_csv("sub.csv", index=False)
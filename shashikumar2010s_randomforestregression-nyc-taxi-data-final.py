import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#my_local_path = "B:/UPX docs/Machine Learning/Project_datasets/Project datasets modified/NYC Taxi Trip/NYC Taxi Trip/"

taxi_data = pd.read_csv('../input/Taxi_new.csv')

taxi_data.head(5)
y = taxi_data["trip_duration"].values



columns = ["hour_of_day", "day_of_month", "month_of_date", "day_of_week_num","distance"]

x = taxi_data[list(columns)].values

x
from sklearn.ensemble import RandomForestRegressor



forest = RandomForestRegressor(max_depth = 10, n_estimators = 100, random_state = 1)
# Fitting the model on Train Data



RF = forest.fit(x, y)
print(RF.score(x, y))
modelPred = RF.predict(x)
list(zip(columns,RF.feature_importances_))
RF.get_params
from sklearn.metrics import mean_squared_error

from math import sqrt
meanSquaredError=mean_squared_error(y, modelPred)

print("MSE:", meanSquaredError)
rootMeanSquaredError = sqrt(meanSquaredError)

print("RMSE:", rootMeanSquaredError)
# Different parameters we want to test



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
# Importing RandomizedSearchCV



from sklearn.model_selection import RandomizedSearchCV
RS_RF = RandomForestRegressor()



# Fitting 3 folds for each of 50 candidates, totalling 300 fits

rf_random = RandomizedSearchCV(estimator = RS_RF, param_distributions = random_grid, 

                               n_iter = 50, cv = 3, verbose=2, random_state=42)
rf_random.fit(x,y)
rf_random.best_params_
best_forest = RandomForestRegressor(max_depth = 90, n_estimators = 90,min_samples_split= 2,min_samples_leaf= 4,

                                    max_features= 'sqrt',bootstrap= True, random_state = 1)
best_RF = best_forest.fit(x, y)
print(best_RF.score(x, y))
best_modelPred = best_RF.predict(x)
best_meanSquaredError=mean_squared_error(y, best_modelPred)

print("MSE:", best_meanSquaredError)
best_rootMeanSquaredError = sqrt(best_meanSquaredError)

print("RMSE:", best_rootMeanSquaredError)
from sklearn import metrics
from sklearn.model_selection import train_test_split



# define a function that accepts a list of features and returns testing RMSE

def train_test_rmse(columns):

    X = taxi_data[columns]

    Y = taxi_data.trip_duration

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123)

    RFreg = RandomForestRegressor(max_depth = 90, n_estimators = 90,min_samples_split= 2,min_samples_leaf= 4,

                                    max_features= 'sqrt',bootstrap= True, random_state = 1)

    RFreg.fit(X_train, y_train)

    y_pred = RFreg.predict(X_test)

    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print (train_test_rmse(['distance', 'hour_of_day','day_of_week_num', 'day_of_month','month_of_date']))
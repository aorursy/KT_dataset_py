import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

import matplotlib.pyplot as plt
%matplotlib inline
cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv('../input/imports-85.data.txt', names=cols)
print(cars.shape)
cars.head()
cars.describe()
continuous_numeric = ['normalized-losses', 'wheel-base', 'length', 'width', 
                      'height', 'curb-weight', 'bore', 'stroke', 'compression-ratio', 
                      'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

numeric_cars = cars[continuous_numeric].copy()
numeric_cars.head()
numeric_cars.isnull().sum()
numeric_cars['normalized-losses'].value_counts()
numeric_cars.replace('?', np.nan, inplace=True)
print("\nMissing values before: \n\n", numeric_cars.isnull().sum(), "\n\n")
numeric_cars.dtypes
to_numeric_cols = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']
numeric_cars[to_numeric_cols] = numeric_cars[to_numeric_cols].astype(float)
numeric_cars.dtypes
numeric_cars.dropna(axis=0, thresh=2, inplace=True)
numeric_cars = numeric_cars.fillna(numeric_cars.mean())
print("\nMissing values after: \n\n", numeric_cars.isnull().sum(), "\n")
normalized_cars = (numeric_cars - numeric_cars.min())/(numeric_cars.max() - numeric_cars.min())
#normalized_cars = np.abs((numeric_cars - numeric_cars.mean())/numeric_cars.std())
normalized_cars['price'] = numeric_cars['price']
print(normalized_cars.shape)
normalized_cars.head()
# Univariate model
def knn_train_test_uni(feature, target_column, df, k):
    
    # Randomize order of rows in data frame.
    np.random.seed(1)
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Split the dataset
    train_set = rand_df.iloc[0:int(len(rand_df)/2)]
    test_set = rand_df.iloc[int(len(rand_df)/2):]
    
    # Train
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(train_set[[feature]], train_set[target_column])
    
    # Predict
    predictions = knn.predict(test_set[[feature]])
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_set[target_column], predictions))
    
    return rmse

k_values = [1, 3, 5, 7, 9]
rmse_uni = {}
current_rmse = []
target_column = 'price'

for feature in continuous_numeric[0:-1]:
    for k in k_values:
        current_rmse.append(knn_train_test_uni(feature, target_column, normalized_cars, k))
        
    rmse_uni[feature] = current_rmse
    current_rmse = []

rmse_uni
fig, ax = plt.subplots(1)

for key, values in rmse_uni.items():
    ax.plot(k_values, values, label=key)
    ax.set_xlabel('k value')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE for Each Training Column\nvs. k value')
    ax.tick_params(top="off", left="off", right="off", bottom='off')
    ax.legend(bbox_to_anchor=(1.5, 1), prop={'size': 11})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
# Multivariate model
def knn_train_test(features, target_column, df, k):
    
    # Randomize order of rows in data frame.
    np.random.seed(1)
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Split the dataset
    train_set = rand_df.iloc[0:int(len(rand_df)/2)]
    test_set = rand_df.iloc[int(len(rand_df)/2):]
    
    # Train
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(train_set[features], train_set[target_column])
    
    # Predict
    predictions = knn.predict(test_set[features])
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_set[target_column], predictions))
    
    return rmse

avg_rmse = {}

for key, values in rmse_uni.items():
    avg_rmse[key] = np.mean(values)

avg_rmse = pd.Series(avg_rmse)
avg_rmse.sort_values()
features = {
        'best_2': ['highway-mpg', 'curb-weight'],
        'best_3': ['highway-mpg', 'curb-weight', 'horsepower'],
        'best_4': ['highway-mpg', 'curb-weight', 'horsepower', 'width'],
        'best_5': ['highway-mpg', 'curb-weight', 'horsepower', 'width', 'city-mpg'],
        'best_6': ['highway-mpg', 'curb-weight', 'horsepower', 'width', 'city-mpg', 'length']
    } 

rmse_multi = {}
target_column = 'price'
k = 5

for key, value in features.items():
    rmse_multi[key] = knn_train_test(value, target_column, normalized_cars, k)
    
pd.Series(rmse_multi).sort_values()
top_models = {
        'best_2': ['highway-mpg', 'curb-weight'],
        'best_3': ['highway-mpg', 'curb-weight', 'horsepower'],
        'best_6': ['highway-mpg', 'curb-weight', 'horsepower', 'width', 'city-mpg', 'length']
    } 

k_values = list(range(1, 26))
rmse_multi_k = {}
rmse_current = []

for key, value in top_models.items():
    for k in k_values:
        rmse_current.append(knn_train_test(value, target_column, normalized_cars, k))
        
    rmse_multi_k[key] = rmse_current
    rmse_current = []
    
print(rmse_multi_k)
# Returns a dict with the min value of every key's list and its index the list
def min_key_value(dictionary):
    min_values = {}
    for k, v in dictionary.items():
        min_values[k] = [min(v), v.index(min(v))]
        
    return min_values
        
best_k = min_key_value(rmse_multi_k)
print(best_k)

# Plot results
fig, ax = plt.subplots(1)

for key, values in rmse_multi_k.items():
    ax.plot(k_values, values, label=key)
    ax.set_xlabel('k value')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE for Top 3 Models vs. k value\n Test/Train Validation')
    ax.tick_params(top="off", left="off", right="off", bottom='off')
    ax.legend(bbox_to_anchor=(1.5, 1), prop={'size': 11})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
def knn_cross_validation(features, target_column, df, k): 
    knn = KNeighborsRegressor(n_neighbors=k)
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    mses = cross_val_score(knn, df[features], df[target_column], scoring='neg_mean_squared_error', cv=kf)
    avg_rmse = np.mean(np.sqrt(np.absolute(mses)))
    
    return avg_rmse

features = {
        'best_2': ['highway-mpg', 'curb-weight'],
        'best_3': ['highway-mpg', 'curb-weight', 'horsepower'],
        'best_4': ['highway-mpg', 'curb-weight', 'horsepower', 'width'],
        'best_5': ['highway-mpg', 'curb-weight', 'horsepower', 'width', 'city-mpg'],
        'best_6': ['highway-mpg', 'curb-weight', 'horsepower', 'width', 'city-mpg', 'length']
    } 

rmse_multi = {}
target_column = 'price'
k = 5

for key, value in features.items():
    rmse_multi[key] = knn_cross_validation(value, target_column, normalized_cars, k)
    
pd.Series(rmse_multi).sort_values()


top_models = {
        'best_3': ['highway-mpg', 'curb-weight', 'horsepower'],
        'best_4': ['highway-mpg', 'curb-weight', 'horsepower', 'width'],
        'best_5': ['highway-mpg', 'curb-weight', 'horsepower', 'width', 'city-mpg']
    } 

k_values = list(range(1, 26))
rmse_multi_k = {}
rmse_current = []

for key, value in top_models.items():
    for k in k_values:
        rmse_current.append(knn_cross_validation(value, target_column, normalized_cars, k))
        
    rmse_multi_k[key] = rmse_current
    rmse_current = []
    
print(rmse_multi_k)
best_k = min_key_value(rmse_multi_k)
print(best_k)

# Plot results
fig, ax = plt.subplots(1)

for key, values in rmse_multi_k.items():
    ax.plot(k_values, values, label=key)
    ax.set_xlabel('k value')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE for Top 3 Models vs. k value\n 10-Fold Cross Validation')
    ax.tick_params(top="off", left="off", right="off", bottom='off')
    ax.legend(bbox_to_anchor=(1.5, 1), prop={'size': 11})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

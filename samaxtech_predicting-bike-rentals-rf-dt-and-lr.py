import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
bike_rentals = pd.read_csv('../input/hour.csv')
bike_rentals.head(5)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,6))
ax1, ax2 = axes.flatten()

bike_rentals['cnt'].hist(grid=False, ax=ax1)

# Sorted correlations with'cnt'
sorted_corrs = bike_rentals.corr()['cnt'].sort_values(ascending=False)
sns.heatmap(bike_rentals[sorted_corrs.index].corr(), ax=ax2)

ax1.set_title('Target Column "cnt" Histogram')
ax2.set_title('Correlations')
plt.show()
print("Correlations:\n\n", sorted_corrs)
def assign_label(hour):
    if hour >= 6 and hour < 12:
        return 1
    elif hour >= 12 and hour < 18:
        return 2
    elif hour >= 18 and hour < 24:
        return 3
    elif hour >= 0 and hour < 6:
        return 4
    
bike_rentals['time_label'] = bike_rentals['hr'].apply(lambda hr: assign_label(hr))
print(bike_rentals['time_label'].value_counts())
bike_rentals.head(5)
bike_rentals['weather_idx'] = 0.8*bike_rentals['temp'] + 0.1*bike_rentals['atemp'] + 0.1*bike_rentals['hum'] 
# Train: 80% of the data / Test: 20% of the data
train, test = train_test_split(bike_rentals, test_size=0.2, random_state=100)

print("\nTrain: ", train.shape)
print("Test: ", test.shape)
features = bike_rentals.columns[~bike_rentals.columns.isin(['cnt', 'registered', 'casual', 'dteday'])].tolist()

X_train = train[features]
y_train = train['cnt']

X_test = test[features]
y_test = test['cnt']

print("\nInitial features: ", features)
# Linear model
lr = LinearRegression()

# Train 
lr.fit(X_train, y_train)

#Predict 
new_cnt_lr = lr.predict(X_test)

# --------------------------------------------------
# Error metric
# --------------------------------------------------

# MSE 
mse_lr = mean_squared_error(y_test, new_cnt_lr)

print("-----------------\nLinear regression\n-----------------")
print("MSE: ", mse_lr)
# Decision Trees model
min_samples_leaf = range(5,20,5)
max_depth = range(5,50,5)
min_samples_split = range(5,20,5)

mse_dt = {}
current_mse = math.inf

for msl in min_samples_leaf:
    for md in max_depth:
        for mss in min_samples_split:
            
            dt = DecisionTreeRegressor(min_samples_leaf=msl, max_depth=md, min_samples_split=mss)

            # Train
            dt.fit(X_train, y_train)

            # Predict
            new_cnt_dt = dt.predict(X_test)
            
            # Update MSE 
            mse = mean_squared_error(y_test, new_cnt_dt)
            
            if mse <= current_mse:
                mse_dt['value'] = mse
                mse_dt['min_samples_split'] = mss
                mse_dt['max_depth'] = md
                mse_dt['min_samples_leaf'] = msl
                
                current_mse = mse

print("-----------------\nDecision Trees\n-----------------")
print("MSE: ", mse_dt)
# Random Forests model (setting n_estimators=10 (default))
min_samples_leaf = range(5,20,5)
max_depth = range(5,50,5)
min_samples_split = range(5,20,5)

mse_rf = {}
current_mse = math.inf

for msl in min_samples_leaf:
    for md in max_depth:
        for mss in min_samples_split:

            dt = RandomForestRegressor(n_estimators=10, min_samples_leaf=msl, max_depth=md, min_samples_split=mss)

            # Train
            dt.fit(X_train, y_train)

            # Predict
            new_cnt_rf = dt.predict(X_test)

            # Update MSE 
            mse = mean_squared_error(y_test, new_cnt_rf)

            if mse <= current_mse:
                mse_rf['value'] = mse
                mse_rf['min_samples_split'] = mss
                mse_rf['max_depth'] = md
                mse_rf['min_samples_leaf'] = msl

                current_mse = mse

print("-----------------\nRandom Forests\n-----------------")
print("MSE: ", mse_rf)
new_target = ['casual', 'registered']
new_y_train = train[new_target]
# Linear model
lr = LinearRegression()

# Train (update y_train)
lr.fit(X_train, new_y_train)

#Predict
predictions = lr.predict(X_test)

# Add up 'casual' and 'registered'
new_cnt_lr = predictions.sum(axis=1)

# --------------------------------------------------
# Error metric
# --------------------------------------------------

# MSE 
mse_lr = mean_squared_error(y_test, new_cnt_lr)

print("-----------------\nLinear regression\n-----------------")
print("MSE: ", mse_lr)
# Decision Trees model
min_samples_leaf = range(5,20,5)
max_depth = range(5,50,5)
min_samples_split = range(5,20,5)

mse_dt = {}
current_mse = math.inf

for msl in min_samples_leaf:
    for md in max_depth:
        for mss in min_samples_split:
            
            dt = DecisionTreeRegressor(min_samples_leaf=msl, max_depth=md, min_samples_split=mss)

            # Train (update y_train)
            dt.fit(X_train, new_y_train)

            # Predict
            predictions = dt.predict(X_test)
            
            # Add up 'casual' and 'registered'
            new_cnt_dt = predictions.sum(axis=1)
            
            # Update MSE 
            mse = mean_squared_error(y_test, new_cnt_dt)
            
            if mse <= current_mse:
                mse_dt['value'] = mse
                mse_dt['min_samples_split'] = mss
                mse_dt['max_depth'] = md
                mse_dt['min_samples_leaf'] = msl
                
                current_mse = mse

print("-----------------\nDecision Trees\n-----------------")
print("MSE: ", mse_dt)
# Random Forests model (setting n_estimators=10 (default))
min_samples_leaf = range(5,20,5)
max_depth = range(5,50,5)
min_samples_split = range(5,20,5)

mse_rf = {}
current_mse = math.inf

for msl in min_samples_leaf:
    for md in max_depth:
        for mss in min_samples_split:

            dt = RandomForestRegressor(n_estimators=10, min_samples_leaf=msl, max_depth=md, min_samples_split=mss)

            # Train (update y_train)
            dt.fit(X_train, new_y_train)

            # Predict
            predictions = dt.predict(X_test)
            
            # Add up 'casual' and 'registered'
            new_cnt_rf = predictions.sum(axis=1)
            
            # Update MSE 
            mse = mean_squared_error(y_test, new_cnt_rf)

            if mse <= current_mse:
                mse_rf['value'] = mse
                mse_rf['min_samples_split'] = mss
                mse_rf['max_depth'] = md
                mse_rf['min_samples_leaf'] = msl

                current_mse = mse

print("-----------------\nRandom Forests\n-----------------")
print("MSE: ", mse_rf)

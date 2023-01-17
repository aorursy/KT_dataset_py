# Data processing
import numpy as np
import pandas as pd
import datetime as dt

# Visualization libaries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Machine Learning
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Read data
#n = 55000000 # Number of total rows
#s = 100000 # Desired sample size
#skip = sorted(np.random.choice(range(n), n-s, replace=False))
#skip[0] = 1
#train = pd.read_csv('../input/train.csv', skiprows=skip, header=0)
train = pd.read_csv('../input/train.csv', nrows=10_000_000)
test = pd.read_csv('../input/test.csv')

train.dtypes
# Let's start by checking for NaN values
print('Sum of NaN values for each column')
print(train.isnull().sum())

# It seems like we lost some data for the dropoff. There are several ways of handling this, but I just go with removing the rows.
train = train.dropna()
print('Sum of NaN values for each column after dropping NaN')
print(train.isnull().sum())
# Let's have a look at the data
train.describe()
# So there seem to be a lot of outliers.

# Manually picking reasonable levels until I find a smarter way
train = train.loc[(train['fare_amount'] > 0) & (train['fare_amount'] < 200)]
train = train.loc[(train['pickup_longitude'] > -300) & (train['pickup_longitude'] < 300)]
train = train.loc[(train['pickup_latitude'] > -300) & (train['pickup_latitude'] < 300)]
train = train.loc[(train['dropoff_longitude'] > -300) & (train['dropoff_longitude'] < 300)]
train = train.loc[(train['dropoff_latitude'] > -300) & (train['dropoff_latitude'] < 300)]
#train = train.loc[train[columns_to_select] < ]
# Let's assume taxa's can be mini-busses as well, so we select up to 8 passengers.
train = train.loc[train['passenger_count'] <= 8]
train.describe()
print('Sum of NaN values for each column')
print(test.isnull().sum())
combine = [test, train]
for dataset in combine:
    # Distance is expected to have an impact on the fare
    dataset['longitude_distance'] = dataset['pickup_longitude'] - dataset['dropoff_longitude']
    dataset['latitude_distance'] = dataset['pickup_latitude'] - dataset['dropoff_latitude']
    
    # Straight distance
    dataset['distance_travelled'] = (dataset['longitude_distance'] ** 2 + dataset['latitude_distance'] ** 2) ** .5
    dataset['distance_travelled_sin'] = np.sin((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5)
    dataset['distance_travelled_cos'] = np.cos((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5)
    dataset['distance_travelled_sin_sqrd'] = np.sin((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5) ** 2
    dataset['distance_travelled_cos_sqrd'] = np.cos((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5) ** 2
    
    # Haversine formula for distance
    # Haversine formula:	a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    R = 6371e3 # Metres
    phi1 = np.radians(dataset['pickup_latitude'])
    phi2 = np.radians(dataset['dropoff_latitude'])
    phi_chg = np.radians(dataset['pickup_latitude'] - dataset['dropoff_latitude'])
    delta_chg = np.radians(dataset['pickup_longitude'] - dataset['dropoff_longitude'])
    a = np.sin(phi_chg / 2) ** .5 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2) ** .5
    c = 2 * np.arctan2(a ** .5, (1-a) ** .5)
    d = R * c
    dataset['haversine'] = d
    
    # Bearing
    # Formula:	θ = atan2( sin Δλ ⋅ cos φ2 , cos φ1 ⋅ sin φ2 − sin φ1 ⋅ cos φ2 ⋅ cos Δλ )
    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    dataset['bearing'] = np.arctan2(y, x)
    
    # Maybe time of day matters? Obviously duration is a factor, but there is no data for time arrival
    # Features: hour of day (night vs day), month (some months may be in higher demand) 
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'])
    dataset['hour_of_day'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['week'] = dataset.pickup_datetime.dt.week
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['day_of_year'] = dataset.pickup_datetime.dt.dayofyear
    dataset['week_of_year'] = dataset.pickup_datetime.dt.weekofyear
    

# Remove rows with zero distance from training set
train = train.loc[train['haversine'] != 0]
train = train.dropna()

    
train.head()
print('Train data: Sum of NaN values for each column')
print(train.isnull().sum())
print('Test data: Sum of NaN values for each column')
print(test.isnull().sum())
# So we have a lot of NaN values for the Haversine for the test set.
# This is probably because python cannot work with to short distances.
# This is not good, since there's a lot more NaN values than actual values.
# So maybe Haversine isn't such a good feature afterall?
# If used, it should be fixed somehow. Ideas?
# One way could be by using the median or mean. However, this is not accurate.
median = test['haversine'].median()
test['haversine'] = test['haversine'].fillna(median)
# Let's check how the features correlate
colormap = plt.cm.RdBu
plt.figure(figsize=(20,20))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
# Let's drop all the irrelevant features
train_features_to_keep = ['haversine', 'fare_amount']
train.drop(train.columns.difference(train_features_to_keep), 1, inplace=True)

test_features_to_keep = ['haversine', 'key']
test.drop(test.columns.difference(test_features_to_keep), 1, inplace=True)
# Let's experiment with different models.
# Process:
# 1. Get predictions using Linear Regression, Random Forest and XGBoost.
# 2. Check how prediction correlate.
# 3. Take a weighted average of predictions
# 4. Submit and fingers crossed \X/
# Step 1:
# Let's combine the training set again
x_train = train.drop('fare_amount', axis=1)
y_train = train['fare_amount']
x_test = test.drop('key', axis=1)

# Set up the models.
# Linear Regression Model
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x_train, y_train)
regr_pred = regr.predict(x_test)

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
rfr_pred = rfr.predict(x_test)
# Let's prepare the test set
x_pred = test.drop('key', axis=1)

# Let's run XGBoost and predict those fares!
x_train,x_test,y_train,y_test = train_test_split(train.drop('fare_amount',axis=1),train.pop('fare_amount'),random_state=123,test_size=0.2)

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=matrix_train,num_boost_round=200, 
                    early_stopping_rounds=20,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)
xgb_pred = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)
regr_pred, rfr_pred, xgb_pred
# Assigning weights. More precise models gets higher weight.
regr_weight = 1
rfr_weight = 1
xgb_weight = 3
prediction = (regr_pred * regr_weight + rfr_pred * rfr_weight + xgb_pred * xgb_weight) / (regr_weight + rfr_weight + xgb_weight)
prediction
# Add to submission
submission = pd.DataFrame({
        "key": test['key'],
        "fare_amount": prediction.round(2)
})

submission.to_csv('sub_fare.csv',index=False)
submission

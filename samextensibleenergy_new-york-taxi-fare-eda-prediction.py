# Data processing
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date


# Visualization libaries
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
%matplotlib inline

# Machine Learning
from sklearn.model_selection import train_test_split
import xgboost as xgb
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv', nrows = 1_000_000)

print ("Train Dataset: Rows, Columns: ", train.shape)
print ("Test Dataset: Rows, Columns: ", test.shape)
print ("Glimpse of Train Dataset: ")
train.head(n=3)
test.head(n=3)
# Lets Check out the missing value columns
print ("Top Columns having missing values")
missmap = train.isnull().sum().to_frame().sort_values(0, ascending = False)
missmap.head()
train = train.dropna() # dropping the NAN values
print ("Number of missing values after dropping NaNs")
missmap = train.isnull().sum().to_frame().sort_values(0, ascending = False)
missmap.head()
print ("Summary of Train Dataset: ")
train.describe()
# Plotting the histogram of counts of passengers,
# binning it into 208 parts from the fact that the maximum value is 208
x = train['passenger_count']
plt.hist(x, bins=208, color = 'yellow', edgecolor = 'black')
plt.ylabel('Counts')
plt.xlabel('Number of Passengers per trip')
plt.show()
# selecting rows with 5 passengers or less only
train = train.loc[train['passenger_count'] <= 5]
train.describe()
# Statistically Outliers are considered mean values - 3 times the standard deviation,
# but considering the Latitude and longitude to be decimal point sensitive, 
#I would personally stringent the factor to 2 

columns_to_select = [ 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']

for column in columns_to_select:
    train = train.loc[(train[column] > (train[column].mean() - train[column].std() * 2 )) & (train[column] < (train[column].mean() + train[column].std() * 2 ))]

train.describe()
train = train.loc[train['fare_amount'] >= 0]
train.describe()
# 33 data points have been eliminated
# renaming our Cleaned Dataset
train_clean = train
   # Haversine formula:	a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
R = 6371e3 # Metres
phi1 = np.radians( train_clean['pickup_latitude'])
phi2 = np.radians( train_clean['dropoff_latitude'])

delta_phi = np.radians( train_clean['pickup_latitude'] - train_clean['dropoff_latitude'])
delta_lambda = np.radians( train_clean['pickup_longitude'] - train_clean['dropoff_longitude'])
a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
c = 2 * np.arctan2(a ** .5, (1-a) ** .5)
d = R * c
train_clean['haversine'] = 0.000621371 *d # converting Meters to miles
train_clean.head(n=10)    
train_clean['haversine'].describe()
   # Lets de-construct the date, time, weekday and weekend factors.
train_clean['pickup_datetime'] = pd.to_datetime(train_clean['pickup_datetime'])
train_clean['hour_of_day'] = train_clean.pickup_datetime.dt.hour
train_clean['day'] = train_clean.pickup_datetime.dt.day
train_clean['week'] = train_clean.pickup_datetime.dt.weekday
train_clean['month'] = train_clean.pickup_datetime.dt.month
train_clean['day_of_year'] = train_clean.pickup_datetime.dt.dayofyear
train_clean['week_of_year'] = train_clean.pickup_datetime.dt.weekofyear
train_clean.head()

g_yr = train_clean.groupby('day_of_year')
yr = g_yr.count()
yr = yr.reset_index()

ax =  yr.set_index('day_of_year')[['key']].plot.bar(figsize=(200, 40), legend=True, fontsize=12, 
                 edgecolor = 'black', 
                 alpha=0.49, 
                 color = 'hotpink') 

ax.set_xlabel("Days of Year", fontsize=30)
ax.set_ylabel("Counts of Trips", fontsize=30)
plt.show()
g_wk = train_clean.groupby('week_of_year')
wk = g_wk.count()
wk = wk.reset_index()

ax =  wk.set_index('week_of_year')[['key']].plot.bar(figsize=(40, 20), legend=True, fontsize=24, 
                 edgecolor = 'black', 
                 alpha=0.79, 
                 color = 'purple') 

ax.set_xlabel("Weeks of Year", fontsize=30)
ax.set_ylabel("Counts of Trips", fontsize=30)
plt.show()
g_mn = train_clean.groupby('month')
mn = g_mn.count()
mn = mn.reset_index()

ax =  mn.set_index('month')[['key']].plot.bar(figsize=(40, 20), legend=True, fontsize=24, 
                 edgecolor = 'black', 
                 alpha=0.79, 
                 color = 'darkcyan') 

ax.set_xlabel("Month of Year", fontsize=30)
ax.set_ylabel("Counts of Trips", fontsize=30)
plt.show()
g_hr = train_clean.groupby('hour_of_day')
hr = g_hr.count()
hr = hr.reset_index()

ax =  hr.set_index('hour_of_day')[['key']].plot.bar(figsize=(40, 20), legend=True, fontsize=24, 
                 edgecolor = 'black', 
                 alpha=0.79, 
                 color = 'orange') 

ax.set_xlabel("Hour of Day", fontsize=30)
ax.set_ylabel("Counts of Trips", fontsize=30)
plt.show()
g_wk_ = train_clean.groupby('week')
wk_= g_wk_.count()
wk_= wk_.reset_index()

ax =  wk_.set_index('week')[['key']].plot.bar(figsize=(40, 20), legend=True, fontsize=24, 
                 edgecolor = 'black', 
                 alpha=0.79, 
                 color = 'salmon') 

ax.set_xlabel("Weekday", fontsize=30)
ax.set_ylabel("Counts of Trips", fontsize=30)
plt.show()
d = train_clean[[ 'fare_amount', 'haversine', 'passenger_count', 'hour_of_day', 'week', 'month' ]]
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

colormap = plt.cm.RdBu
plt.figure(figsize=(8,8))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(d.corr(),linewidths=0.5,vmax=1.0, mask=mask,
            square=True, cmap=colormap, linecolor='white', annot=True, cbar_kws={"shrink": .5})
test.describe()

   # Haversine formula:	a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
R = 6371e3 # Metres
phi1 = np.radians( test['pickup_latitude'])
phi2 = np.radians( test['dropoff_latitude'])

delta_phi = np.radians( test['pickup_latitude'] - test['dropoff_latitude'])
delta_lambda = np.radians( test['pickup_longitude'] - test['dropoff_longitude'])
a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
c = 2 * np.arctan2(a ** .5, (1-a) ** .5)
d = R * c
test['haversine'] = 0.000621371 *d # converting Meters to miles


# generating Similar columns for Test set too
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
test['hour_of_day'] = test.pickup_datetime.dt.hour
test['day'] = test.pickup_datetime.dt.day
test['week'] = test.pickup_datetime.dt.weekday
test['month'] = test.pickup_datetime.dt.month
test['day_of_year'] = test.pickup_datetime.dt.dayofyear
test['week_of_year'] = test.pickup_datetime.dt.weekofyear

test.head(n=3)    
test.head()
# Let's drop all the irrelevant features
# train_features_to_keep = ['haversine', 'hour_of_day', 'fare_amount']
#  train_clean.drop(train_clean.columns.difference(train_features_to_keep), 1, inplace=True)
train_model_ = train_clean
test_model_ = test
# test_features_to_keep = ['haversine', 'hour_of_day', 'key']
# test.drop(test.columns.difference(test_features_to_keep), 1, inplace=True)
test_model_.head()
train_model= train_model_[['haversine', 'hour_of_day', 'fare_amount']]
test_model = test_model_[['haversine', 'hour_of_day', 'key']]
train_model.head()
x_pred = test_model.drop('key', axis=1)

# Let's run XGBoost and predict those fares!
x_train,x_test,y_train,y_test = train_test_split(train_model.drop('fare_amount',axis=1),train_model.pop('fare_amount'),random_state=123,test_size=0.2)

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=matrix_train,num_boost_round=200, 
                    early_stopping_rounds=10,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)
prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)
# Add to submission
submission = pd.DataFrame({
        "key": test['key'],
        "fare_amount": prediction.round(2)
})

submission.to_csv('predicted_fare.csv',index=False)
submission
# from sklearn.svm import SVR
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# train_model= train_model_[['haversine', 'hour_of_day', 'fare_amount']]
# test_model = test_model_[['haversine', 'hour_of_day', 'key']]
# test_model.head()
# X_pred = test_model.drop('key', axis=1)

# # Let's run XGBoost and predict those fares!
# X = train_model.drop('fare_amount',axis=1)
# y = train_model.pop('fare_amount')

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# # Scale the data to be between -1 and 1
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# # Establish a model
# model = SVR(C=1, cache_size=500, epsilon=1, kernel='rbf')

# # Train the model - this will take a minute
# m_fit = model.fit(X_train, y_train)
# pre = m_fit.predict(X_pred)


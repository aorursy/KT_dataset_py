import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import statsmodels as sm
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv', nrows= 10_000_000)
df_train.pickup_datetime = df_train.pickup_datetime.str.slice(0,16)
df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'], utc=True, format = '%Y-%m-%d %H:%M')
df_train.info()
df_train.head()
df_train.describe()
# Filter out negative fare amount and maximum passenger_count
print("Old Size Before Filter: %d" %(len(df_train)))
df_train = df_train[(df_train.fare_amount >=0) & (df_train.passenger_count <=10)]
print("New Size After Filter: %d" %(len(df_train)))
# Let's plot histogram of fare_amount to see its distribution across data.

df_train[df_train.fare_amount < 100].fare_amount.hist(bins=100, figsize=(10,5))
plt.xlabel("Fare in USD $")
plt.title("Fare Amount Distribution")
print(df_train.isnull().sum())
print("Old Size : %d" %(len(df_train)))
df_train=df_train.dropna(axis = 0)
print("New Size: %d" %(len(df_train)))
# Filtering out off boundary points. Boundary of New York City is (-75, -73, 40, 42)
def NYC(df):
    boundary_filter = (df.pickup_longitude >= -75) & (df.pickup_longitude <= -73) & \
                      (df.pickup_latitude >= 40) & (df.pickup_latitude <= 42) & \
                      (df.dropoff_longitude >= -75) & (df.dropoff_longitude <= -73) & \
                      (df.dropoff_latitude >= 40) & (df.dropoff_latitude <= 42)
    df = df[boundary_filter]
    return df
print('Old size: %d' % len(df_train))
df_train = NYC(df_train)
print('New size: %d' % len(df_train))
def distance_between_pickup_dropoff(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    d = np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)
    return d
# Extracting Features 

df_train['hour'] = df_train.pickup_datetime.dt.hour
df_train['day'] = df_train.pickup_datetime.dt.day
df_train['month'] = df_train.pickup_datetime.dt.month
df_train['year'] = df_train.pickup_datetime.dt.year
df_train.drop('pickup_datetime', axis =1, inplace = True)

# Creating actual_distance column as measure of manhattan distance

df_train['actual_distance'] = distance_between_pickup_dropoff(df_train.pickup_latitude, df_train.pickup_longitude,
                                                             df_train.dropoff_latitude, df_train.dropoff_longitude)
# Let's check how our new data set looks like.

df_train.head()
# Here, i am bounding the longitude and latitude values to get clear and zoomed plot.
df_plot = df_train[(df_train.pickup_longitude >= -74.1)&(df_train.pickup_longitude <= -73.8) & (df_train.pickup_latitude >=40.6)
                  & (df_train.pickup_latitude <=40.9)]
# In scatter plot arguments c = 'r' is "color red", s= 0.01 is "size of dots" and alpha = 0.5 is "opacity of dots"
fig, ax = plt.subplots(1, 1, figsize=[10,10])
ax.scatter(df_plot.pickup_longitude[:3_00_000], df_plot.pickup_latitude[:3_00_000],c = 'r', s= 0.01,alpha=0.5)
# Let's zoom little more

zoomed_data =  df_train[(df_train.pickup_longitude >= -74.02)&(df_train.pickup_longitude <= -73.95) & (df_train.pickup_latitude >=40.7)
                  & (df_train.pickup_latitude <=40.80)]
fig, ax = plt.subplots(1, 1, figsize=[10,10])
ax.scatter(zoomed_data.pickup_longitude[:3_00_000], zoomed_data.pickup_latitude[:3_00_000],c = 'b', s= 0.01,alpha=0.5)
# Let's create feature vector
# We do not want trip involving 0 passenger_count
filt = (df_train.passenger_count > 0) & (df_train.fare_amount < 250)
features = ['passenger_count','hour','year','day','month','actual_distance']
for f in features:
    related = df_train.fare_amount.corr(df_train[f])
    print("%s: %f" % (f,related))
final_features =['year','hour','actual_distance','passenger_count']
X = df_train[filt][final_features].values # Feature Vector
Y = df_train[filt]['fare_amount'].values # Target Variable
X.shape, Y.shape
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
# Splitting data set into train and test 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
regressor = LinearRegression()
metric = 'neg_mean_squared_error'
scores = cross_val_score(regressor, X_test, y_test, cv = 10, scoring = metric)
scores
np.sqrt(np.abs(scores))
np.sqrt(np.abs(scores.mean()))
regressor.fit(X_train, y_train)
y_train_pred = regressor.predict(X_train)
def error(y, y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))
rmse = error(y_train, y_train_pred)
rmse
y_test_pred = regressor.predict(X_test)
rmse = error(y_test, y_test_pred)
rmse
from sklearn.linear_model import Lasso
alphas =[1e-5,1e-3, 1e-2, 0.02, 0.04,0.08,0.1]
for alpha in alphas:
    lasso = Lasso(alpha = alpha)
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    rmse = error(y_train, y_train_pred)
    print("alpha : {%.5f} RMSE : {%.9f}" %(alpha,rmse))
lasso = Lasso(alpha = 0.01)
lasso.fit(X_train, y_train)
y_test_pred = lasso.predict(X_test)
rmse = error(y_test, y_test_pred)
rmse
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(max_depth = 17)
reg.fit(X_train, y_train)
y_trn_pred = reg.predict(X_train)
rmse = error(y_train, y_trn_pred)
rmse
y_tst_pred = reg.predict(X_test)

rmse = error(y_test, y_tst_pred)
rmse
df_test =  pd.read_csv('../input/test.csv')
df_test.head()
df_test.pickup_datetime = df_test.pickup_datetime.str.slice(0,16)

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'], utc=True, format = '%Y-%m-%d %H:%M')
# Extracting Features for test set

df_test['hour'] = df_test.pickup_datetime.dt.hour
df_test['day'] = df_test.pickup_datetime.dt.day
df_test['month'] = df_test.pickup_datetime.dt.month
df_test['year'] = df_test.pickup_datetime.dt.year
df_test.drop('pickup_datetime', axis =1, inplace = True)

# Creating actual_distance column as measure of manhattan distance

df_test['actual_distance'] = distance_between_pickup_dropoff(df_test.pickup_latitude, df_test.pickup_longitude,
                                                             df_test.dropoff_latitude, df_test.dropoff_longitude)
X_test = df_test[final_features].values
y_pred_test_set = reg.predict(X_test)
submission =  pd.DataFrame({'key': df_test.key, 'fare_amount': y_pred_test_set},columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)
submission

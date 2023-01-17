import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
print(os.listdir("../input"))


train_df =  pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = 1000000, parse_dates=["pickup_datetime"])
test_df =  pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv',  parse_dates=["pickup_datetime"])
#only the first 1000000 rows are used to reduce running time
train_df.dtypes
#data cleaning part1
# Find null values
#Identify null values
print(train_df.isnull().sum())
#Drop rows with null values
print('Old size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train_df))
#data cleaning part2
#drop off data with extremely long distance between departure and arrival
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
add_travel_vector_features(train_df)
plot = train_df.iloc[:20000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
print('Old size: %d' % len(train_df))
train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]
print('New size: %d' % len(train_df))
plot = train_df.iloc[:20000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
#data cleaning part3
#drop off data with strange longitude and latitude
print('Old size: %d' % len(train_df))
plot = train_df.iloc[:20000].plot.scatter('dropoff_longitude', 'dropoff_latitude')
plot = train_df.iloc[:20000].plot.scatter('pickup_longitude', 'pickup_latitude')
train_df = train_df[(train_df.dropoff_longitude < -70) & (train_df.dropoff_latitude > 35) & (train_df.pickup_longitude < -70) & (train_df.pickup_latitude > 35)]
train_df = train_df[(train_df.dropoff_longitude > -80) & (train_df.dropoff_latitude < 45) & (train_df.pickup_longitude > -80) & (train_df.pickup_latitude < 45)]
plot = train_df.iloc[:20000].plot.scatter('dropoff_longitude', 'dropoff_latitude')
plot = train_df.iloc[:20000].plot.scatter('pickup_longitude', 'pickup_latitude')
print('New size: %d' % len(train_df))

#Visualisation part1
# plot histogram of fare
train_df.fare_amount.hist(bins=100, figsize=(14,3))
plt.xlabel('fare $USD')
plt.title('Histogram');
#data cleaning part4
#drop off data with fare below the smallest possible fare
print('Old size: %d' % len(train_df))
plot = train_df.iloc[:20000].plot.scatter('fare_amount', 'fare_amount')
train_df = train_df[(train_df.fare_amount > 3)]
print('New size: %d' % len(train_df))
plot = train_df.iloc[:20000].plot.scatter('fare_amount', 'fare_amount')
#Visualisation part2
# plot amount of records each day in train and test set
train_df['pickup_datetime'] = pd.to_datetime(train_df.pickup_datetime)
test_df['pickup_datetime'] = pd.to_datetime(test_df.pickup_datetime)
train_df.loc[:, 'pickup_date'] = train_df['pickup_datetime'].dt.date
test_df.loc[:, 'pickup_date'] = test_df['pickup_datetime'].dt.date
plt.plot(train_df.groupby('pickup_date').count()[['key']], 'o-', label='train')
plt.plot(test_df.groupby('pickup_date').count()[['key']], 'o-', label='test')
plt.title('Train and test period complete overlap.')
plt.legend(loc=0)
plt.ylabel('number of records')
plt.show()
#Visualisation part3
# plot amount of records each weakday in train and test set
train_df['pickup_datetime'] = pd.to_datetime(train_df.pickup_datetime)
test_df['pickup_datetime'] = pd.to_datetime(test_df.pickup_datetime)
train_df.loc[:, 'pickup_weekday'] = train_df['pickup_datetime'].dt.weekday
test_df.loc[:, 'pickup_weekday'] = test_df['pickup_datetime'].dt.weekday
plt.plot(train_df.groupby('pickup_weekday').count()[['key']], 'o-', label='train')
plt.plot(test_df.groupby('pickup_weekday').count()[['key']], 'o-', label='test')
plt.title('Train and test period complete overlap.')
plt.legend(loc=0)
plt.ylabel('number of records')
plt.show()
#We found that the need of taxi on weekdays is higher than that on weekends.
#While in test dataset, the chosen records are balanced according to the day of the week.
#Visualisation part4
# plot amount of records each hour in train and test set
train_df['pickup_datetime'] = pd.to_datetime(train_df.pickup_datetime)
test_df['pickup_datetime'] = pd.to_datetime(test_df.pickup_datetime)
train_df.loc[:, 'pickup_hour'] = train_df['pickup_datetime'].dt.hour
test_df.loc[:, 'pickup_hour'] = test_df['pickup_datetime'].dt.hour
plt.plot(train_df.groupby('pickup_hour').count()[['key']], 'o-', label='train')
plt.plot(test_df.groupby('pickup_hour').count()[['key']], 'o-', label='test')
plt.title('Train and test period complete overlap.')
plt.legend(loc=0)
plt.ylabel('number of records')
plt.show()
#From the plot we found that in train set the record of taxi rides reach the peak at around 19:00 while there are not much need of taxi at around 5:00 everyday,
#At the same tiem the chosen of test set based on the time period is balanced.
import lightgbm as lgb
test_test = pd.read_csv('../input/dataafterclean/testAfterClean.csv')
train = pd.read_csv('../input/dataafterclean/trainAfterClean.csv')
val= pd.read_csv('../input/dataafterclean/validationAfterClean.csv')
test= pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv')

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values,
                                          train['pickup_longitude'].values, 
                                          train['dropoff_latitude'].values, 
                                          train['dropoff_longitude'].values)
test_test.loc[:, 'direction'] = bearing_array(test_test['pickup_latitude'].values,
                                         test_test['pickup_longitude'].values, 
                                         test_test['dropoff_latitude'].values, 
                                         test_test['dropoff_longitude'].values)
val.loc[:, 'direction'] = bearing_array(val['pickup_latitude'].values,
                                         val['pickup_longitude'].values, 
                                         val['dropoff_latitude'].values, 
                                         val['dropoff_longitude'].values)


from sklearn.decomposition import PCA
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values,
                    val[['pickup_latitude', 'pickup_longitude']].values,
                    val[['dropoff_latitude', 'dropoff_longitude']].values,
                    test_test[['pickup_latitude', 'pickup_longitude']].values,
                    test_test[['dropoff_latitude', 'dropoff_longitude']].values))

pca = PCA().fit(coords)
train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
val['pickup_pca0'] = pca.transform(val[['pickup_latitude', 'pickup_longitude']])[:, 0]
val['pickup_pca1'] = pca.transform(val[['pickup_latitude', 'pickup_longitude']])[:, 1]
val['dropoff_pca0'] = pca.transform(val[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
val['dropoff_pca1'] = pca.transform(val[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
test_test['pickup_pca0'] = pca.transform(test_test[['pickup_latitude', 'pickup_longitude']])[:, 0]
test_test['pickup_pca1'] = pca.transform(test_test[['pickup_latitude', 'pickup_longitude']])[:, 1]
test_test['dropoff_pca0'] = pca.transform(test_test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
test_test['dropoff_pca1'] = pca.transform(test_test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
X_train = train.drop(['fare_amount','pickup_latitude', 'pickup_longitude','dropoff_latitude', 'dropoff_longitude'], axis=1)
Y_train = train["fare_amount"]
X_test = val.drop(['fare_amount','pickup_latitude', 'pickup_longitude','dropoff_latitude', 'dropoff_longitude'], axis=1)
Y_test = val["fare_amount"]
test_test = test_test.drop(['pickup_latitude', 'pickup_longitude','dropoff_latitude', 'dropoff_longitude'], axis=1)

Y_test = Y_test.reset_index().drop('index',axis = 1)
Y_train = Y_train.reset_index().drop('index',axis = 1)
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=Y_train)
dvalid = xgb.DMatrix(X_test, label=Y_test)
dtest = xgb.DMatrix(test_test)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
xgb_pars = {'min_child_weight': 1, 'eta': 0.1, 'colsample_bytree': 0.9, 
            'max_depth': 10,
'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 1000, watchlist, early_stopping_rounds=40,
      maximize=False, verbose_eval=1)
print('Modeling RMSLE %.5f' % model.best_score)
xgb.plot_importance(model, max_num_features=10, height=0.7)


pred = model.predict(dtest)
submission = pd.concat([test['key'], pd.DataFrame(pred)], axis=1)
submission.columns = ['key','fare_amount']
submission.to_csv("sub_xgb.csv", index=False)

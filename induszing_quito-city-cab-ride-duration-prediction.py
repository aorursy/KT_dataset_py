# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, ElasticNet, Lasso



from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

from sklearn.preprocessing import StandardScaler

from bayes_opt import BayesianOptimization



%matplotlib inline

plt.rcParams['figure.figsize'] = [16, 10]



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
uio = pd.read_csv('../input/uio_clean.csv', header=0, sep=',', encoding="Latin1")

uio.head()
uio['pickup_datetime'] = pd.to_datetime(uio['pickup_datetime'], format='%Y/%m/%d %H:%M:%S')

uio['dropoff_datetime'] = pd.to_datetime(uio['dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')



#checked a sample of file in notepad also to verify the datetime format. 
print(uio['pickup_datetime'].dt.time.max()) #There seems to be some data issue with this file as the pickup times are only between 01:00:01hours and 12:59:59 hours

print(uio['pickup_datetime'].dt.time.min())
uio.info()

#The datetime values are successfully converted
uio.describe()
uio['vendor_id'].unique()
uio['store_and_fwd_flag'].unique() #store and forward flag has only one value - so it's not useful for modeling purpose
np.percentile(uio['trip_duration'], np.arange(0,101))
np.percentile(uio['pickup_latitude'], np.arange(0,101))
uio = uio[(uio['dropoff_latitude'] >= -0.32628809) & (uio['dropoff_latitude'] <= -0.07806344)]   #remove the outliers from dropoff latitude

uio.shape
uio = uio[(uio['pickup_latitude'] >= -0.32628809) & (uio['pickup_latitude'] <= -0.07806344)]   #remove the outliers from pickup latitude

uio.shape

np.percentile(uio['dist_meters']/1000, np.arange(90,101)) #percentile distribution of trip distance in km. Based on this the upper limit

#for ditastance is chosen as  50km
uio['dist_meters'] = uio['dist_meters']/1000

uio = uio[uio['dist_meters'] <= 50]  

uio.shape
np.percentile(uio['trip_duration']/60, np.arange(0,101))
uio['trip_duration'] =  uio['trip_duration']/60

uio.shape
uio = uio[(uio['trip_duration'] >= 0.5) & (uio['trip_duration'] <=350. )] #trip duration must be between 1 min and 350 minutes

uio.shape
np.percentile(uio['trip_duration'], np.arange(0,101))   #veriy the values of trip duration in minutes
uio['wait_sec'] = uio['wait_sec']/60

np.percentile(uio['wait_sec'], np.arange(0,101))  #distrubution of waiting time in minutes. 

#Based on this we will drop any records with more than 100 munutes of waiting tmie
uio = uio[uio['wait_sec'] <= 100]

uio.shape
uio[uio['trip_duration'] <= uio['wait_sec']].count() #Trip duration should not be less than waiting time
uio =uio[uio['trip_duration'] > uio['wait_sec']]

uio.shape
uio[uio['trip_duration'] <= uio['wait_sec']].count()
np.percentile(uio['trip_duration']- (uio['dropoff_datetime'] - uio['pickup_datetime']).dt.seconds/60, np.arange(0,101))

# There are some records with more than 2 minutes difference - these will need to be removed
uio = uio[np.abs(uio['trip_duration']- (uio['dropoff_datetime'] - uio['pickup_datetime']).dt.seconds/60) <= 2]

uio.shape
np.percentile((uio['trip_duration']- (uio['dropoff_datetime'] - uio['pickup_datetime']).dt.seconds/60), np.arange(0,101)) 

#verify the distribution of difference between trip duration and drop offf - pickup date times
np.percentile(uio['pickup_longitude'], np.arange(0,101))
np.percentile(uio['dropoff_longitude'], np.arange(0,101))
uio.isna().sum()  #check if any NaN / nulls present
#In the absence of any information about how data is captured for pickup and drop off datetime and trip duration, 

#we can't make any assumption regarding which one is more accurate. So, dropping records where the time difference 

#is more than 2 minute. We are using 2 minutes here based on the percentile distribution shown in previous section 

#for this metric
uio.shape
uio.describe()
plt.hist(uio['trip_duration'].values, bins=200)

plt.show()
plt.hist(np.log1p(uio['trip_duration'].values + 1), bins=200)

plt.show()
vendor_freq = uio.groupby(['vendor_id'])['id'].count().reset_index()

vendor_freq
plt.bar(x=vendor_freq['vendor_id'], height=vendor_freq['id'])

plt.title("Distribution of different vendors")

plt.xlabel("Vendor Name")

plt.ylabel("# Trips")

plt.show()
def ride_length_group(distance):

    distance = distance 

    if distance <= 5 : grp = 1

    elif distance <= 10 : grp = 2

    elif distance <=15 : grp = 3

    elif distance <=20 : grp = 4

    elif distance <=25 : grp = 5

    elif distance <=30 : grp = 6

    elif distance <=35 : grp= 7

    elif distance <=40: grp= 8

    elif distance <=45: grp= 9

    else : grp=10

    return grp



def minute_group(minute):

    if minute <= 15 : grp = 1

    elif minute <= 30 : grp = 2

    elif minute <=45 : grp = 3

    else : grp=4

    return grp
uio['ride_length_grp'] = uio['dist_meters'].apply(ride_length_group)
uio['ride_month'] = uio['pickup_datetime'].dt.month

uio['ride_year'] = uio['pickup_datetime'].dt.year

uio['ride_day'] = uio['pickup_datetime'].dt.day

uio['ride_hour'] = uio['pickup_datetime'].dt.hour

uio['ride_minute'] = uio['pickup_datetime'].dt.minute

uio['day_of_week'] = uio['pickup_datetime'].dt.dayofweek



uio['season'] = uio['ride_month'].apply(lambda x: 1 if x >= 6 and x <= 9 else 0) #Quito has two seasons only as per https://en.wikipedia.org/wiki/Climate_of_Ecuador

# Since Quito lies on equator, there is not much difference in daylight duration across the year



uio['vendor_flg'] = uio['vendor_id'].apply(lambda x: 1 if x.lower()=='quito' else 0)

uio['minute_grp'] = uio['ride_minute'].apply(minute_group)



uio['diff_longitude'] = np.round(uio['dropoff_longitude'] - uio['pickup_longitude'], decimals=4)

uio['diff_latitude'] = np.round(uio['dropoff_latitude'] - uio['pickup_latitude'], decimals=4)

uio.head()





# https://www.officeholidays.com/countries/ecuador/2017.php - holidays list of Ecuador

holidays = ['2016-01-01', '2016-02-08', '2016-02-09', '2016-03-25', '2016-03-27', 

            '2016-05-01', '2016-05-27', '2016-07-24', '2016-08-10', '2016-10-09', 

            '2016-11-02', '2016-11-03', '2016-12-06', '2016-12-25', 

            '2017-01-01', '2017-02-27', '2017-02-28', '2017-04-14', '2017-04-16', 

            '2017-05-01', '2017-05-24', '2017-07-24', '2017-08-10', '2017-10-09', 

            '2017-11-02', '2017-11-03', '2017-12-06', '2017-12-25']

holidays = pd.to_datetime(holidays)

# was the day a public holiday?





uio['holiday'] = 1*(pd.to_datetime(uio['pickup_datetime'].dt.date).isin(holidays))



uio['pickup_latitude'] = np.round(uio['pickup_latitude'], decimals =4)

uio['dropoff_latitude'] = np.round(uio['dropoff_latitude'], decimals =4)

uio['pickup_longitude'] = np.round(uio['pickup_longitude'], decimals =4)

uio['dropoff_longitude'] = np.round(uio['dropoff_longitude'], decimals =4)



uio['trip_duration_log'] = np.log1p(uio['trip_duration'] + 1)

#wait_sec and dropoff_time should not be used for prediction as that can cause leakage



uio['average_speed'] = uio['dist_meters'] / (uio['trip_duration']/60)    # in km per hour



uio.head()
uio['holiday'].sum()
x = uio.groupby(['ride_month'])['id'].count() # There is some seasonality as the months of May to Jul and Nov-Dec have higher rides count

sns.barplot(x.index, x.values)
uio['ride_month_gr'] = uio['ride_month'].apply(lambda x: 1 if x in (1,6,7) else 0)

x = uio.groupby(['ride_month_gr'])['id'].count() # There is some seasonality as the months of May to Jul and Nov-Dec have higher rides count

sns.barplot(x.index, x.values)

plt.show()
x = uio.groupby(['ride_year'])['id'].count() # There are more rides in 2017

sns.barplot(x.index, x.values)

plt.show()
x = uio.groupby(['ride_hour'])['id'].count() # There is some seasonality based on the time of the day

sns.barplot(x.index, x.values)

plt.show()
uio['ride_hour_gr'] = uio['ride_hour'].apply(lambda x: 1 if x in (6,7,8,9) else 0)  #flag set to 1 if rides are booked in these hours
x = uio.groupby(['day_of_week'])['id'].count() # There is some seasonality as the months of May to Jul and Nov-Dec have higher rides count

sns.barplot(x.index, x.values)
uio['day_of_week_gr'] = uio['day_of_week'].apply(lambda x: 1 if x in (0,1,2,3,4) else 0) # There is some seasonality as the months of May to Jul and Nov-Dec have higher rides count

x = uio.groupby(['day_of_week_gr'])['id'].count()

sns.barplot(x.index, x.values)
import math

def bearing_array(lat1, lng1, lat2, lng2):

    """bearing:

    horizontal angle between direction of an object and another object"""

    AVG_EARTH_RADIUS = 6371  # in km

    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))
uio.loc[:,'bearing'] = bearing_array(uio['pickup_latitude'].values, uio['pickup_longitude'].values, 

                               uio['dropoff_latitude'].values, uio['dropoff_longitude'].values)

uio.head()
f, axes = plt.subplots(3,2,figsize=(17,10), sharey=True)



dat = uio[uio['season']==1].groupby(['ride_hour'])['id'].count()

fig1 = sns.barplot(dat.index, dat.values, ax=axes[0,0])

#fig1.set_ylim = 3000



dat = uio[uio['season']==0].groupby(['ride_hour'])['id'].count()

fig2 = sns.barplot(dat.index, dat.values, ax=axes[0,1])

#fig2.set_ylim = 3000



dat = uio[uio['season']==1].groupby(['day_of_week'])['id'].count()

fig1 = sns.barplot(dat.index, dat.values, ax=axes[1,0])

#fig1.set_ylim = 3000



dat = uio[uio['season']==0].groupby(['day_of_week'])['id'].count()

fig2 = sns.barplot(dat.index, dat.values, ax=axes[1,1])



dat = uio[uio['season']==1].groupby(['ride_month'])['id'].count()

fig1 = sns.barplot(dat.index, dat.values, ax=axes[2,0])

#fig1.set_ylim = 3000



dat = uio[uio['season']==0].groupby(['ride_month'])['id'].count()

fig2 = sns.barplot(dat.index, dat.values, ax=axes[2,1])
uio = uio.drop(columns = ['vendor_id', 'id', 'dropoff_datetime', 'store_and_fwd_flag', 'pickup_datetime' ], axis=1)

uio.head()
uio.describe()
np.percentile(uio['average_speed'], np.arange(0,101))
uio = uio[uio['average_speed'] <= 150]

uio.shape
uio.describe()
dat = uio

#dat = pd.concat([uio, ride_year, ride_month, ride_day, ride_hour, minute_grp], axis=1)

#dat.head()
y = np.log1p(dat['wait_sec'] + 1)

x = dat.drop(['wait_sec'], axis = 1)

plt.hist(y, bins=200)

plt.show()
train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size = 0.2, shuffle=True, random_state = 65).copy()
print("Shape of X train", train_X.shape)

print("Shape of Y train", train_Y.shape)

print("Shape of X test", test_X.shape)

print("Shape of Y test", test_Y.shape)
coordinates = np.vstack((train_X[['pickup_latitude', 'pickup_longitude']].values, train_X[['dropoff_latitude', 'dropoff_longitude']].values))
scaler = StandardScaler()

coordinates_std = scaler.fit_transform(coordinates)

clustering = MiniBatchKMeans(n_clusters=70, random_state=203, batch_size=10000)

model = clustering.fit(coordinates_std)
X_Tr = train_X.copy().reset_index().drop('index', axis=1)

Y_Tr = train_Y.copy().reset_index().drop('index', axis=1)

X_Te = test_X.copy().reset_index().drop('index', axis=1)

Y_Te = test_Y.copy().reset_index().drop('index', axis=1)



X_Tr['kms_pick_cluster'] = pd.Series(model.predict(scaler.fit_transform(X_Tr[['pickup_latitude', 'pickup_longitude']])))

X_Tr['kms_drop_cluster'] = pd.Series(model.predict(scaler.fit_transform(X_Tr[['dropoff_latitude', 'dropoff_longitude']])))

X_Te['kms_pick_cluster'] = pd.Series(model.predict(scaler.fit_transform(X_Te[['pickup_latitude', 'pickup_longitude']])))

X_Te['kms_drop_cluster'] = pd.Series(model.predict(scaler.fit_transform(X_Te[['dropoff_latitude', 'dropoff_longitude']])))
X_Tr.head()
X_Te.head()


print(X_Tr.shape)

print(Y_Tr.shape)

print(X_Te.shape)

print(Y_Te.shape)



print(X_Tr['kms_pick_cluster'].isna().sum())

print(X_Tr['kms_drop_cluster'].isna().sum())

print(X_Te['kms_pick_cluster'].isna().sum())

print(X_Te['kms_drop_cluster'].isna().sum())
fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharex=True, sharey=True)

axes[0].scatter(X_Tr['pickup_latitude'], X_Tr['pickup_longitude'], c=X_Tr['kms_pick_cluster'], alpha=0.3, lw = 0, s=20, cmap='Spectral')

axes[0].set_title('Pickup Locations Cluster')

axes[0].set_xlabel('Pickup Latitude')

axes[0].set_ylabel('Pickup Longitude')

axes[0].set_ylim([-78.6, -78.3])



axes[1].scatter(X_Tr['dropoff_latitude'], X_Tr['dropoff_longitude'], c=X_Tr['kms_drop_cluster'], alpha=0.3, lw = 0, s=20, cmap='Spectral')

axes[1].set_title('DropOff Locations Cluster')

axes[1].set_xlabel('Pickup Latitude')

axes[1].set_ylabel('Pickup Longitude')

plt.show()
X_Tr.head()
d_Tr = xgb.DMatrix(X_Tr, label=Y_Tr)

d_Te = xgb.DMatrix(X_Te)
def xgb_evaluate(max_depth, gamma,min_child_weight,max_delta_step,subsample,colsample_bytree, eta):

    params = {'eval_metric': 'rmse',

                  'max_depth': int(max_depth),

                  'subsample': subsample,

                  'eta': eta,

                  'gamma': gamma,

                  'colsample_bytree': colsample_bytree,   

                  'min_child_weight': min_child_weight ,

                  'max_delta_step':max_delta_step

                 }

    # Use cross validation to avoid over fitting

    cv_result = xgb.cv(params, d_Tr, num_boost_round=1000, nfold=3, metrics = "rmse", early_stopping_rounds=10, seed=3113)    

    print("Number of Trees", cv_result.shape[0])

        

    # Returning negative of RMSE since Bayesian optimization only knows how to maximize

    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]
%%time

xgb_bo = BayesianOptimization(xgb_evaluate, {

    'eta': (0.01, 0.1),

    'max_depth': (2, 6),

    'gamma': (0, 5),

    'min_child_weight': (1, 20),

    'max_delta_step': (0, 10),

    'subsample': (0.2, 1),

    'colsample_bytree' :(0.2, 0.8)})



# Use the expected improvement acquisition function to handle negative numbers

# Optimally needs quite a few more initiation points and number of iterations

xgb_bo.maximize(init_points=5, n_iter=50, acq='ei')
params = xgb_bo.max['params']

print(params)
params['max_depth'] = int(params['max_depth'])
# Train a new model with the best parameters from the search

model2 = xgb.train(params, d_Tr, num_boost_round=650)



# Predict on testing and training set

y_pred_xgb = model2.predict(d_Te)

y_train_pred_xgb = model2.predict(d_Tr)
# Report testing and training RMSE

print('Test error:', np.sqrt(mean_squared_error(Y_Te, y_pred_xgb)))

print('Train error:', np.sqrt(mean_squared_error(Y_Tr, y_train_pred_xgb)))

sns.distplot(y_pred_xgb, label='y_hat')

sns.distplot(Y_Te, label='y')

plt.legend()

sns.despine()

plt.tight_layout();


wait_sec_train = (np.expm1(Y_Tr)*60).round()

np.quantile(wait_sec_train, [0, 0.05, 0.5, 0.95, 0.99])
wait_sec_true = (np.expm1(Y_Te)*60).round()

wait_sec_predicted = (np.expm1(y_pred_xgb)*60).round()

wait_sec_predicted
wait_sec_true = np.array(wait_sec_true).T[0]

residual = wait_sec_predicted - wait_sec_true.T

wait_sec_true
residual
fig = plt.figure(num=None, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')



plt.subplot(1, 2, 1)

ax1 = sns.distplot(residual)

ax1.set_title('Distribution of residuals\n')

ax1.set_xlabel('')



plt.subplot(1, 2, 2)

ax2 = sns.boxplot(residual, showfliers=False)

ax2.set_xlabel('')



sns.despine()

plt.tight_layout()
print("Mean absolute error:", np.abs(residual).mean())
# feature importance

fig =  plt.figure(figsize = (12,8))

axes = fig.add_subplot(111)

xgb.plot_importance(model2,ax = axes,height =0.5)

sns.despine()

plt.tight_layout()
featuresImp = model2.get_score(importance_type='gain')

print(featuresImp)
grid_search = GridSearchCV( 

                              estimator= ElasticNet(),

                              param_grid={

                              'alpha':[0.001,0.003, 0.01, 0.03,0.05,0.08, 0.1, 0.12, 0.15, 0.17, 0.2, 0.21, 0.22, 0.23, 02.24, 0.25],

                              'l1_ratio': [0.1, 0.25, 0.2, 0.25, 0.3,0.35,0.4,0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90]

                          },

          scoring="neg_mean_squared_error",  

          cv=KFold(n_splits=3,shuffle=True,random_state=42))   

        

grid_search.fit(X_Tr, Y_Tr)

CVed_model = grid_search.best_estimator_

print(grid_search.best_params_)
y_tr_pred_EN = CVed_model.predict(X_Tr)

rmse = np.sqrt(((y_tr_pred_EN - Y_Tr.values.ravel())**2).mean())

print("The train error is: ",rmse)



y_pred_EN = CVed_model.predict(X_Te)

rmse = np.sqrt(((y_pred_EN - Y_Te.values.ravel())**2).mean())

print("The train error is: ",rmse)



# The performance of the Elastic Net is slightly worse than xgboost while using RMSE as comparison
y_tr_pred_EN = CVed_model.predict(X_Tr)

mae = np.abs((y_tr_pred_EN - Y_Tr.values.ravel()))

print("The train error is: ",mae.mean())



y_pred_EN = CVed_model.predict(X_Te)

mae = np.abs((Y_Te.values.ravel() - y_pred_EN))

print("The test error is: ",mae.mean())
y_pred = y_pred_xgb*0.75 + 0.25*y_pred_EN

rmse = np.sqrt(((y_pred - Y_Te.values.ravel())**2).mean())

print("The test error is: ",rmse)
sns.distplot(y_pred, label='y_hat')

sns.distplot(Y_Te, label='y')

plt.legend()

sns.despine()

plt.tight_layout();
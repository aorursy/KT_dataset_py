#ignoring warnings

import warnings

warnings.filterwarnings('ignore')



#data structures

import numpy as np

import pandas as pd



#visualizatons

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style("darkgrid")



#ML packages will add later at model development

#loading all the input data provided

trainset = pd.read_csv("../input/train.csv")

testset = pd.read_csv("../input/test.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")
#manually having a look at datasets

print ("Training Set: \n\n",trainset.head(), "\n\n\n\nTesting Set: \n\n",testset.head(), "\n\n\n\nSample Submission: \n\n",sample_submission.head())
trainset.info() #alter: use trainset.columns for column names and use trainset.dtypes for dtypes
print ("Training set: ")

print ("No of rows: ",trainset.shape[0], "\t\tNo of cols: ",trainset.shape[1])

print ("\nTesting set: ")

print ("No of rows: ",testset.shape[0], "\t\tNo of cols: ",testset.shape[1])
#checking uniqueness

total_rows = trainset.shape[0]

unique_rows = trainset.id.value_counts().shape[0]

if unique_rows == total_rows:

    print (" Training set Consistent." )

else:

    print (" Training set Not Consistent!")



total_rows = testset.shape[0]

unique_rows = testset.id.value_counts().shape[0]

if unique_rows == total_rows:

    print (" Testing set Consistent." )

else:

    print (" Testing set Not Consistent!")
#training and testing are unique?

if len(np.intersect1d(trainset.id.values,testset.id.values))==0:

    print ("Both are Distinct.")

else:

    print ("Both contain few same values!")
#checking for categorical and numerical attributes

cat_fields = [col for col in trainset.columns if trainset.dtypes[col]==object]

num_fields = [col for col in trainset.columns if trainset.dtypes[col]!=object]

print ("Categorical Attributes: ",cat_fields)

print ("\nContinuous Attributes: ",num_fields)
#checking for missing vlaues

no_missing_values = trainset.isnull().sum().sum()

print ("Trainset has ",no_missing_values," missing values")

no_missing_values = testset.isnull().sum().sum()

print ("Testset has ",no_missing_values," missing values")
#convert store_fwd_flag from Y/N to 0/1

trainset["store_and_fwd_flag"] = 1 * (trainset.store_and_fwd_flag.values == 'Y')

testset["store_and_fwd_flag"] = 1 * (testset.store_and_fwd_flag.values =='Y')
trainset['vendor_id'].describe()
trainset['store_and_fwd_flag'].describe()
trainset['passenger_count'].describe()
#describing target variable

trainset['trip_duration'].describe()
by_vid = trainset.groupby('vendor_id')

by_vid['trip_duration'].describe()
by_vid_flag = trainset.groupby(['vendor_id','store_and_fwd_flag'])

by_vid_flag['trip_duration'].describe()
by_pass_count = trainset.groupby('passenger_count')

by_pass_count['trip_duration'].describe()
#transforming pickup and dropoff time to pandas datetime

trainset['pickup_datetime']=pd.to_datetime(trainset['pickup_datetime'])

trainset['dropoff_datetime']=pd.to_datetime(trainset['dropoff_datetime'])



testset['pickup_datetime']=pd.to_datetime(testset['pickup_datetime'])
#visualizing for checking outliers

plt.figure(figsize=(10,10))

plt.scatter(x=trainset['pickup_longitude'].values,y=trainset['pickup_latitude'].values, marker='^',s=1,alpha=.3)

plt.xlim([-74.1,-73.7])

plt.ylim([40.6, 40.9])

plt.axis('off')

plt.show()
def RemoveOutliers(df,cols,n_sigma): # keep only instances that are within n_sigma in columns cols

    new_df = df.copy()

    for col in cols:

        new_df = new_df[np.abs(new_df[col]-new_df[col].mean())<=(n_sigma*new_df[col].std())]

    print('%i instances have been removed' %(df.shape[0]-new_df.shape[0]))

    return new_df
#cleaning 1

clean_att = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']

trainset_clean = RemoveOutliers(trainset,clean_att,5)

testset = RemoveOutliers(testset,clean_att,5)
#dropping dropoff date time

trainset_clean  = trainset.drop('dropoff_datetime',axis=1)

trainset_clean.head(1)
#transforming target variable-> feature1

trainset_clean['log_trip_duration'] = np.log(trainset_clean['trip_duration']+1)

trainset_clean.head(2)
#plot before cleaning trip_duration

plt.figure(figsize=(8,6))



plt.hist(trainset_clean['log_trip_duration'], bins=100)

plt.xlabel('Trip duration (log)')

plt.ylabel('events')



plt.show()
#cleaning 2

clean_att = ['log_trip_duration']

trainset_clean = RemoveOutliers(trainset_clean,clean_att,5)
#plot after cleaning trip_duration

plt.figure(figsize=(8,6))



plt.hist(trainset_clean['log_trip_duration'], bins=100)

plt.xlabel('Trip duration (log)')

plt.ylabel('events')



plt.show()
#feature2

trainset_clean['trip_duration_hr'] = trainset_clean['trip_duration']/3600

trainset_clean.head(2)
#featureset 3-6

def pu_datetime_feature(df):

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    df['pu_hour'] = df['pickup_datetime'].dt.hour

    df['pu_day'] = df['pickup_datetime'].dt.dayofyear

    df['pu_wday'] = df['pickup_datetime'].dt.dayofweek

    df['pu_month'] = df['pickup_datetime'].dt.month
pu_datetime_feature(trainset_clean)

pu_datetime_feature(testset)



trainset_clean.head(n=2)
testset.head(2)
#pickup_hour

sns.countplot(x='pu_hour',data=trainset_clean,hue='vendor_id')

plt.tight_layout()
sns.distplot(trainset_clean['pu_hour'])

plt.tight_layout()
byHour = trainset_clean.groupby('pu_hour').count()['trip_duration']

byHour.plot()

plt.tight_layout()
#pickup_day

sns.countplot(x='pu_day',data=trainset_clean,hue='vendor_id')

plt.tight_layout()
sns.distplot(trainset_clean['pu_day'])

plt.tight_layout()
byDay = trainset_clean.groupby('pu_day').count()['trip_duration']

byDay.plot()

plt.tight_layout()
#pickup_weekday

sns.countplot(x='pu_wday',data=trainset_clean,hue='vendor_id')

plt.tight_layout()
sns.distplot(trainset_clean['pu_wday'])

plt.tight_layout()
byWday = trainset_clean.groupby('pu_wday').count()['trip_duration']

byWday.plot()

plt.tight_layout()
#pickup_month

sns.countplot(x='pu_month',data=trainset_clean,hue='vendor_id')

plt.tight_layout()
sns.distplot(trainset_clean['pu_month'])

plt.tight_layout()
byMonth = trainset_clean.groupby('pu_month').count()['trip_duration']

byMonth.plot()

plt.tight_layout()
#passenger count

sns.countplot(x='passenger_count',data=trainset_clean)

plt.tight_layout()
#store and fwd flag

sns.countplot(x='store_and_fwd_flag',data=trainset_clean)

plt.tight_layout()
#distance calculation

#haversine distance

def haversine(lat1,lon1,lat2,lon2):

    lat1,lon1,lat2,lon2 = map(np.radians,(lat1,lon1,lat2,lon2))

    dlon = np.abs(lon2-lon1)

    dlat = np.abs(lat2-lat1)

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2

    c = 2*np.arcsin(np.sqrt(a))

    R = 6371 #radius of earth in km

    d = R*c

    return d
#manhattan distance

def manhattan(lat1, long1, lat2, long2):

    a = Haversine(lat1, long1, lat1, long2)

    b = Haversine(lat1, long2, lat2, long2)

    return a + b
trainset_clean['haversine'] = haversine(trainset['dropoff_latitude'],trainset['dropoff_longitude'],trainset['pickup_latitude'],trainset['pickup_longitude'])  

trainset_clean['manhattan'] = haversine(trainset['dropoff_latitude'],trainset['dropoff_longitude'],trainset['pickup_latitude'],trainset['pickup_longitude'])



testset['haversine'] = haversine(testset['dropoff_latitude'],testset['dropoff_longitude'],testset['pickup_latitude'],testset['pickup_longitude'])  

testset['manhattan'] = haversine(testset['dropoff_latitude'],testset['dropoff_longitude'],testset['pickup_latitude'],testset['pickup_longitude'])

trainset_clean.head(2)
testset.head(2)
#average speed in km/hr

trainset_clean['avg_speed'] = trainset_clean['haversine']/trainset_clean['trip_duration_hr']

trainset_clean.head(2)
trainset_clean['avg_speed'].describe()
fig, ax = plt.subplots(ncols=3, sharey=True)

ax[0].plot(trainset_clean.groupby('pu_hour').mean()['avg_speed'], 'bo-', lw=2, alpha=0.7)

ax[1].plot(trainset_clean.groupby('pu_day').mean()['avg_speed'], 'go-', lw=2, alpha=0.7)

ax[2].plot(trainset_clean.groupby('pu_wday').mean()['avg_speed'], 'ro-', lw=2, alpha=0.7)

ax[0].set_xlabel('hour')

ax[1].set_xlabel('day')

ax[2].set_xlabel('weekday')

ax[0].set_ylabel('average speed')

fig.suptitle('Plot of average traffic speed')

plt.show()
def bearing(lat1, lng1, lat2, lng2):

    AVG_EARTH_RADIUS = 6371  # in km

    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))
trainset_clean['direction'] = bearing(trainset['dropoff_latitude'],trainset['dropoff_longitude'],trainset['pickup_latitude'],trainset['pickup_longitude'])  

testset['direction'] = bearing(testset['dropoff_latitude'],testset['dropoff_longitude'],testset['pickup_latitude'],testset['pickup_longitude'])  

trainset_clean.head(2)
testset.head(2)
trainset_clean['center_long'] = (trainset['pickup_longitude']+trainset['dropoff_longitude'])/2  

trainset_clean['center_lat'] = (trainset['pickup_latitude'] + trainset['dropoff_latitude'])/2



testset['center_long'] = (testset['pickup_longitude']+testset['dropoff_longitude'])/2  

testset['center_lat'] = (testset['pickup_latitude'] + testset['dropoff_latitude'])/2
testset.head(2)
trainset_clean.head(2)
#X_train = trainset_clean[['vendor_id','passenger_count', 'pickup_longitude', 'pickup_latitude',

#       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',

#        'pu_hour', 'pu_day' ,'pu_wday', 'pu_month']]

X_train = trainset_clean.drop(['id','pickup_datetime','trip_duration','log_trip_duration','trip_duration_hr','avg_speed'],1)

y_train =trainset_clean['log_trip_duration']

X_test = testset.drop(['id','pickup_datetime'],1)

print (X_train.dtypes)

print (X_test.dtypes)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
lr = LinearRegression()

lr.fit(X_train,y_train)

lr.predict(X_test)[:10]
sns.distplot(lr.predict(X_test))
pd.DataFrame(zip(X_train.columns,lr.coef_), columns = ["features","estimated coefficients"])
#high co-relation between pickup_longitude and trip_duration

plt.scatter(X_train.pickup_longitude,y_train)

plt.show()
plt.scatter(y_train,lr.predict(X_train)) #actual vs predicted

plt.show()
print (X_train.shape)
#cross validation

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=5)
print (X_train.shape)

print (X_test.shape)

print (y_train.shape)

print (y_test.shape)
lr = LinearRegression()

lr.fit(X_train,y_train)
pred_train = lr.predict(X_train)

pred_test = lr.predict(X_test)
print ("Fit a model X_train, and calculate MSE with Y_train:", np.mean((y_train - lr.predict(X_train)) ** 2))

print ("Fit a model X_train, and calculate MSE with X_test, Y_test:", np.mean((y_test - lr.predict(X_test)) ** 2))
plt.scatter(pred_train, pred_train-y_train,c='b',s=40,alpha=0.5)

plt.scatter(pred_test, pred_test-y_test,c='g',s=40)

plt.hlines(y=0,xmin=0,xmax=50)

plt.title("Residual plot using training (blue), test (green) data.")

plt.ylabel("Residuals")

plt.show()
import xgboost as xgb

Xtr, Xv, ytr, yv = train_test_split(X_train, y_train, test_size=0.2, random_state=1987)

dtrain = xgb.DMatrix(Xtr, label=ytr)

dvalid = xgb.DMatrix(Xv, label=yv)

#dtest = xgb.DMatrix(test[feature_names].values)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



#From beluga's kernel

xgb_pars = {'min_child_weight': 50, 'eta': 0.3, 'colsample_bytree': 0.3, 'max_depth': 10,

            'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 60, watchlist, early_stopping_rounds=50,

                  maximize=False, verbose_eval=10)
print('Modeling RMSLE %.5f' % model.best_score)

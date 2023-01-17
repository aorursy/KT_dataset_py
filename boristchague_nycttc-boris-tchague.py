import os 



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn

import matplotlib.mlab as mlab

import warnings

warnings.filterwarnings("ignore")





from sklearn import linear_model

from sklearn.metrics import mean_squared_error, mean_squared_error as MSE

from sklearn.linear_model import LinearRegression, Lasso

from sklearn.model_selection import cross_val_score, cross_val_predict







# Settings

import matplotlib

matplotlib.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = (8.5, 5)

plt.rcParams["patch.force_edgecolor"] = True

pd.set_option('display.float_format', lambda x: '%.3f' % x)

sns.mpl.rc("figure", figsize=(8.5,5))





%matplotlib inline
import os



train = pd.read_csv ('../input/train.csv')

test = pd.read_csv ('../input/test.csv')



train.head(2)
print(train.shape);

train.info()
#### we have no missing value in our data set. 

train.isna().sum()
train.describe().transpose()
train = train[train['passenger_count']>0]

train = train[train['passenger_count']<9]
#Pro-processing Pickup cordinates

train = train[train['pickup_longitude'] <= -73.75]

train = train[train['pickup_longitude'] >= -74.03]

train = train[train['pickup_latitude'] <= 40.85]

train = train[train['pickup_latitude'] >= 40.63]





#Pro-processing dropoff cordinates 

train = train[train['dropoff_longitude'] <= -73.75]

train = train[train['dropoff_longitude'] >= -74.03]

train = train[train['dropoff_latitude'] <= 40.85]

train = train[train['dropoff_latitude'] >= 40.63]
#Pre-processing trip duration 

trip_duration_mean = np.mean(train['trip_duration'])

trip_duration_std = np.std(train['trip_duration'])

train = train[train['trip_duration']<=trip_duration_mean + 2*trip_duration_std]

train = train[train['trip_duration']>= trip_duration_mean - 2*trip_duration_std]
# Confirm removal

train.describe().transpose()
# Pickups

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime']) 



# Drop-offs

train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime']) 
# Confirm changes

train.info()
# Decomposing pickup_datetime variable into date and time

train['pickup_date'] = train['pickup_datetime'].dt.date # Extract date

train['pickup_time'] = train['pickup_datetime'].dt.time # Extract time



# Decomposing dropoff_datetime variable into date and time 

train['dropoff_date'] = train['dropoff_datetime'].dt.date # Extract date

train['dropoff_time'] = train['dropoff_datetime'].dt.time # Extract time







# Additional pickup features

train['pickup_month'] = train['pickup_datetime'].dt.month # Extract month



train['pickup_hour'] = train['pickup_datetime'].dt.hour # Extract hour



train['pickup_weekday'] = train['pickup_datetime'].dt.dayofweek # Extract day of week
# Drop concatentated timestamp columns

train.drop(['pickup_datetime'], axis = 1, inplace = True)

train.drop(['dropoff_datetime'], axis = 1, inplace = True)



# Confirm removal

train.columns
# Differences between dropoff and pickup geocardiante helps to calculate distance covered during a trips



train['dist_long'] = train['pickup_longitude'] - train['dropoff_longitude']

#test['dist_long'] = test['pickup_longitude'] - test['dropoff_longitude']



train['dist_lat'] = train['pickup_latitude'] - train['dropoff_latitude']

#test['dist_lat'] = test['pickup_latitude'] - test['dropoff_latitude']





# Distance covered

train['dist'] = np.sqrt(np.square(train['dist_long']) + np.square(train['dist_lat']))

#test['dist'] = np.sqrt(np.square(test['dist_long']) + np.square(test['dist_lat']))
# Mean distribution

mu = train['trip_duration'].mean()



# Std distribution

sigma = train['trip_duration'].std()

num_bins = 100



# Histogram 

fig = plt.figure(figsize=(8.5, 5))

n, bins, patches = plt.hist(train['trip_duration'], num_bins, normed=1,

                           edgecolor = 'black', lw = 1, alpha = .40)

# Normal Distribution

y = mlab.normpdf(bins, mu, sigma)

plt.plot(bins, y, 'r--', linewidth=2)

plt.xlabel('trip_duration')

plt.ylabel('Probability density')



# Adding a title

plt.title(r'$\mathrm{Trip\ duration\ skewed \ to \ the \ right:}\ \mu=%.3f,\ \sigma=%.3f$'%(mu,sigma))

plt.grid(True)

#fig.tight_layout()

plt.show();





import matplotlib

matplotlib.style.use('fivethirtyeight')



# Create boxplot

plt.figure(figsize=(8.5,5))

passenger_graph = sns.boxplot(x = 'passenger_count', y = 'trip_duration', data = train, 

                          palette = 'gist_rainbow', linewidth = 2.3)



# Customize tick size

passenger_graph.tick_params(axis = 'both', which = 'major', labelsize = 12)





# Customize tick labels of the y-axis

passenger_graph.set_yticklabels(labels = [-10, '0  ', '2000  ', '4000  ', '6000  ', '8000  ', '10000  ','12000 s'])





# Bolding horizontal line at y = 0

passenger_graph.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .70)





# # Adding a title and a subtitle

passenger_graph.text(x =-1.05, y = 13800, s = "Passenger count does not have much effect on trip duration",

               fontsize =15 , weight = 'bold', alpha = .90)

passenger_graph.text(x = -1.05, y = 13000.3, 

               s = 'Median trip times remain similar despite more passengers being aboard',

              fontsize = 14, alpha = .85)

plt.show()



# Trips by Hour and Day of Week

trip_duration_median = train['trip_duration'].median()

plt.figure(figsize=(8.5,5))



pickup_hourday = train.groupby(['pickup_hour','pickup_weekday'])['trip_duration'].median().unstack()

hourday_graph = sns.heatmap(pickup_hourday[pickup_hourday>trip_duration_median], lw = .5, annot = True, cmap = 'GnBu', fmt = 'g',annot_kws = {"size":10} )





# Customize tick labels of the y-axis

hourday_graph.set_xticklabels(labels = ['Mon', 'Tue', 'Wed','Thu','Fri','Sat','Sun'])





# Remove the label of the x-axis

hourday_graph.xaxis.label.set_visible(False)



# # Adding a title and a subtitle

hourday_graph.text(x =-.8, y = 27, s = "Trip durations vary greatly depending on day of week",

               fontsize =20 , weight = 'bold', alpha = .90)



plt.show()
# Box plot of pickups by month

import matplotlib

matplotlib.style.use('fivethirtyeight')



# Create boxplot

plt.figure(figsize=(8.5,5))

month_graph = sns.boxplot(x = 'pickup_month', y = 'trip_duration', data = train, palette = 'gist_rainbow', linewidth = 2.3)



# Remove the label of the x-axis

month_graph.xaxis.label.set_visible(False)

month_graph.yaxis.label.set_visible(False)



month_graph.text(x =-1.05, y = 13800, s = "Month of transaction has minimal effect on trip duration",

               fontsize =20 , weight = 'bold', alpha = .90)

month_graph.text(x = -1.05, y = 13000.3, 

               s = 'Median trip times hover around ~650 seconds throughout the year',

              fontsize = 14, alpha = .85)

plt.show()





# Statistical summary

train.groupby('pickup_month')['trip_duration'].describe().transpose()



# Correlations to trip_duration

corr = train.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

cor_dict = corr['trip_duration'].to_dict()

del cor_dict['trip_duration']

print("List the numerical features in decending order by their correlation with trip_duration:\n")

for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):

    print("{0}: {1}".format(*ele))

    

# Correlation matrix heatmap

corrmat = train.corr()

plt.figure(figsize=(12, 7))



# Number of variables for heatmap

k = 76

cols = corrmat.nlargest(k, 'trip_duration')['trip_duration'].index

cm = np.corrcoef(train[cols].values.T)



# Generate mask for upper triangle

mask = np.zeros_like(cm, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.set(font_scale=1)

sns.heatmap(cm, mask=mask, cbar=True, annot=True, square=True,\

                 fmt='.2f',annot_kws={'size': 12}, yticklabels=cols.values,\

                 xticklabels=cols.values, cmap = 'coolwarm',lw = .1)

plt.show() 
# Encoding Categoric data (converting 'store_and_fwd_flag' to numeric)

train['store_and_fwd_flag'] = train['store_and_fwd_flag'].map({'N':0,'Y':1})
train.drop(columns=['pickup_date','pickup_time','dropoff_date', 'dropoff_time', 'dist_long', 'dist_lat'], axis = 1, inplace = True)
train.head(3)
X = train[['vendor_id', 'passenger_count', 'pickup_longitude',

       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',

       'store_and_fwd_flag','pickup_month', 'pickup_hour',

       'pickup_weekday', 'dist']]



# Target

y = train['trip_duration']
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(

    X, y, test_size = 0.2, random_state = 0)

X_train.shape, y_train.shape, X_val.shape, y_val.shape
#instantiate model

lr = LinearRegression()



# Fit to training data

lr = lr.fit(X_train,y_train);



#Predict

y_pred_lr = lr.predict(X_val)
#cross_validation_score

cvs_lr = np.sqrt(

    -cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))

print(cvs_lr)



mean_cvs_lr = np.mean(cvs_lr)

print(mean_cvs_lr)
# Score It

from sklearn import metrics

print('\nRandom Forest Regression Performance Metrics')

print('R^2 =',metrics.explained_variance_score(y_val,y_pred_lr))

print('MAE',metrics.mean_absolute_error(y_val, y_pred_lr))

print('MSE',metrics.mean_squared_error(y_val, y_pred_lr))

print('RMSE',np.sqrt(metrics.mean_squared_error(y_val, y_pred_lr)))
#model selection 

from sklearn.ensemble import RandomForestRegressor



# Intantiate model 

rf = RandomForestRegressor(n_estimators = 20, n_jobs = -1)



#fit

rf = rf.fit(X_train, y_train)



#Predict

y_pred_rf = rf.predict(X_val)



# crosse validation

cvs_rf = cross_val_score(rf, X_train, y_train, cv=5)

print(cvs_rf)



mean_cvs_rf = np.mean(cvs_rf)

print(mean_cvs_rf)
from sklearn import metrics

print('\nRandom Foresst Performance Metrics')

print('R^2=',metrics.explained_variance_score(y_val,y_pred_rf))

print('MAE:',metrics.mean_absolute_error(y_val,y_pred_rf))

print('MSE:',metrics.mean_squared_error(y_val,y_pred_rf))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_val,y_pred_rf)))

print('RMSLE:',np.sqrt(metrics.mean_squared_log_error(y_val, y_pred_rf)))
# Test data info

test.info()



# Test data shape

print('shape',test.shape)
# Convert timestamps to date objects

test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime) # Pickups



# Delimit pickup_datetime variable 

test['pickup_date'] = test['pickup_datetime'].dt.date # Extract date

test['pickup_time'] = test['pickup_datetime'].dt.time # Extract time



# Additional pickup features

test['pickup_month'] = test['pickup_datetime'].dt.month # Extract month



#train_data['pickup_YYYYMM'] = train_data['pickup_datetime'].apply(lambda x: x.strftime('%Y%m')) # Extract yearmonth

test['pickup_hour'] = test['pickup_datetime'].dt.hour # Extract hour

test['pickup_weekday'] = test['pickup_datetime'].dt.dayofweek # Extract day of week



# Encode categorical variables

test['store_and_fwd_flag'] = test['store_and_fwd_flag'].map({'N':0,'Y':1})





# Differences between dropoff and pickup geocardiante helps to calculate distance covered during a trips

test['dist_long'] = test['pickup_longitude'] - test['dropoff_longitude']

test['dist_lat'] = test['pickup_latitude'] - test['dropoff_latitude']



# Distance covered

test['dist'] = np.sqrt(np.square(test['dist_long']) + np.square(test['dist_lat']))



#drop useless colomns

test.drop(columns=['pickup_datetime', 'pickup_date', 'dist_long', 'dist_lat'], axis = 1, inplace = True)



# Create new matrix of features from test data

X_test= test[['vendor_id', 'passenger_count', 'pickup_longitude',

       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',

       'store_and_fwd_flag','pickup_month', 'pickup_hour',

       'pickup_weekday', 'dist']]



# Feed features into random forest

test_pred= rf.predict(X_test)
submission = pd.DataFrame({'id':test['id'], 'trip_duration': test_pred})

submission.to_csv('mySubmission.csv', index=False)

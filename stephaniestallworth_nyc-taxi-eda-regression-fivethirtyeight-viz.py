# Imports

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.mlab as mlab

import pandas as pd

import seaborn as sns

import sklearn

import warnings

warnings.filterwarnings("ignore")



# Settings

import matplotlib

matplotlib.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = (8.5, 5)

plt.rcParams["patch.force_edgecolor"] = True

pd.set_option('display.float_format', lambda x: '%.3f' % x)

sns.mpl.rc("figure", figsize=(8.5,5))



# Read data

train_data = pd.read_csv('../input/train.csv')



# View data

train_data.head()
# Data Shape

print('Data Shape',train_data.shape)

train_data.info()
# Statistical summary

train_data.describe().transpose()
# Remove passenger_count outliers

train_data = train_data[train_data['passenger_count']>0]

train_data = train_data[train_data['passenger_count']<9]



# train_data = train_data[train_data['pickup_longitude'] <= -73.968285]

# train_data = train_data[train_data['pickup_longitude'] >= -74.0059]

# train_data = train_data[train_data['pickup_latitude'] <= 40.748817]

# train_data = train_data[train_data['pickup_latitude'] >= 40.7128]

# train_data = train_data[train_data['dropoff_longitude'] <= -73.968285]

# train_data = train_data[train_data['dropoff_longitude'] >= -74.0059]

# train_data = train_data[train_data['dropoff_latitude'] <= 40.748817]

# train_data = train_data[train_data['dropoff_latitude'] >= 40.7128]



# Remove coordinate outliers

train_data = train_data[train_data['pickup_longitude'] <= -73.75]

train_data = train_data[train_data['pickup_longitude'] >= -74.03]

train_data = train_data[train_data['pickup_latitude'] <= 40.85]

train_data = train_data[train_data['pickup_latitude'] >= 40.63]

train_data = train_data[train_data['dropoff_longitude'] <= -73.75]

train_data = train_data[train_data['dropoff_longitude'] >= -74.03]

train_data = train_data[train_data['dropoff_latitude'] <= 40.85]

train_data = train_data[train_data['dropoff_latitude'] >= 40.63]



# Remove trip_duration outliers

trip_duration_mean = np.mean(train_data['trip_duration'])

trip_duration_std = np.std(train_data['trip_duration'])

train_data = train_data[train_data['trip_duration']<=trip_duration_mean + 2*trip_duration_std]

train_data = train_data[train_data['trip_duration']>= trip_duration_mean - 2*trip_duration_std]



# Confirm removal

train_data.describe().transpose()
train_data.info()
# Convert timestamps to date objects

train_data['pickup_datetime'] = pd.to_datetime(train_data.pickup_datetime) # Pickups

train_data['dropoff_datetime'] = pd.to_datetime(train_data.dropoff_datetime) # Drop-offs



# Confirm changes

train_data.info()
# Delimit pickup_datetime variable 

train_data['pickup_date'] = train_data['pickup_datetime'].dt.date # Extract date

train_data['pickup_time'] = train_data['pickup_datetime'].dt.time # Extract time



# Delimit dropoff_datetime variables

train_data['dropoff_date'] = train_data['dropoff_datetime'].dt.date # Extract date

train_data['dropoff_time'] = train_data['dropoff_datetime'].dt.time # Extract time



# Additional pickup features

train_data['pickup_month'] = train_data['pickup_datetime'].dt.month # Extract month

# train_data['pickup_month'] = train_data.pickup_datetime.dt.to_period('M') # Extract yearmonth

#train_data['pickup_YYYYMM'] = train_data['pickup_datetime'].apply(lambda x: x.strftime('%Y%m')) # Extract yearmonth

train_data['pickup_hour'] = train_data['pickup_datetime'].dt.hour # Extract hour

train_data['pickup_weekday'] = train_data['pickup_datetime'].dt.dayofweek # Extract day of week



# Drop concatentated timestamp columns

train_data.drop(['pickup_datetime'], axis = 1, inplace = True)

train_data.drop(['dropoff_datetime'], axis = 1, inplace = True)



# Confirm changes

train_data.columns
# Mean distribution

mu = train_data['trip_duration'].mean()



# Std distribution

sigma = train_data['trip_duration'].std()

num_bins = 100



# Histogram 

fig = plt.figure(figsize=(8.5, 5))

n, bins, patches = plt.hist(train_data['trip_duration'], num_bins, normed=1,

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

plt.show()



# Statistical summary

train_data.describe()[['trip_duration']].transpose()
# Feature names

train_data.columns
# Summarize total trips by day

pickups_by_day = train_data.groupby('pickup_date').count()['id']



# Create graph

pickups_graph = pickups_by_day.plot(x = 'pickup_date', y = 'id', figsize = (8.5,5),legend = True)



# Customize tick size

pickups_graph.tick_params(axis = 'both', which = 'major', labelsize = 12)



# Bold horizontal line at y = 0

pickups_graph.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)



# Customize tick labels of the y-axis

#pickups_graph.set_yticklabels(labels = [-10, '2000   ', '4000   ', '6000   ', '8000   ', '10000   '])



# Add an extra vertical line by tweaking the range of the x-axis

pickups_graph.set_xlim(left = '2015-12-31', right = '2016-06-30')



# Remove the label of the x-axis

pickups_graph.xaxis.label.set_visible(False)



# Add signature bar

pickups_graph.text(x = '2015-12-15', # Adjusts left side of signature bar,has to be in same coordiantes as x-axis

               y = -2500, 

               s = '    ©KAGGLE                                          Source: NYC Taxi and Limousine Commission (TLC)   ', # copyright symbol ALT + 0169

              fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')



# Adding a title and a subtitle

pickups_graph.text(x = '2015-12-18', y = 11800,

                   s = "Dramatic drop in total trips in late January or early February",

                   fontsize = 20, weight = 'bold', alpha = .90)



pickups_graph.text(x = '2015-12-18', y = 11000, 

                   s = 'Decline is isolated to a specific day so may be more than just seasonal effects.',

                   fontsize = 14, alpha = .85)

pickups_graph.text(x = '2016-01-27', y = 1500, s = 'What happened?',weight = 0, rotation = 0, backgroundcolor = '#f0f0f0', size = 14)

plt.show()
# Identify where drop occured

train_data.groupby('pickup_date').count()['id'].sort_values(ascending = True)[[0]]
# Create boxplot

plt.figure(figsize=(8.5,5))

vendor_graph = sns.boxplot(x = 'vendor_id', y = 'trip_duration', data = train_data, 

                          palette = 'gist_rainbow', linewidth = 2.3)



# Customize tick labels of the y-axis

vendor_graph.set_yticklabels(labels = [-10, '0  ', '2000  ', '4000  ', '6000  ', '8000  ', '10000  ','12000 s'])



# Bolding horizontal line at y = 0

vendor_graph.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .70)



# Remove the label of the x-axis

vendor_graph.xaxis.label.set_visible(False)

vendor_graph.yaxis.label.set_visible(False)



# Add signature bar

vendor_graph.text(x = -.66, # Adjusts left side of signature bar

               y = -2500,  

               s = '   ©KAGGLE                                                 Source: NYC Taxi and Limousine Commission (TLC)   ', # copyright symbol ALT + 0169

              fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey') 



# # Adding a title and a subtitle

vendor_graph.text(x =-.66, y = 13800, s = "Trip durations are similar between NYC taxi vendors",

               fontsize =20 , weight = 'bold', alpha = .90)

vendor_graph.text(x = -.66, y = 13000.3, 

               s = 'Both have a median trip time ~650 seconds with many outliers',

              fontsize = 14, alpha = .85)

plt.show()



# Statistical summary

train_data.groupby('vendor_id')['trip_duration'].describe()

# Create boxplot

plt.figure(figsize=(8.5,5))

vendor_graph = sns.boxplot(x = 'vendor_id', y = 'trip_duration', data = train_data, 

                          orient = 'v',color = 'lightgrey', linewidth = 2.3)

plt.setp(vendor_graph.artists, alpha = 0.5)



# Create strip plot

sns.stripplot(data = train_data, x = 'vendor_id', y = 'trip_duration', jitter = 1, size = 5,

             edgecolor = 'black', linewidth = .2,palette = 'gist_rainbow_r',hue = 'store_and_fwd_flag')



# Customize tick size

vendor_graph.tick_params(axis = 'both', which = 'major', labelsize = 12)



# Customize tick labels of the y-axis

vendor_graph.set_yticklabels(labels = [-10, '0  ', '2000  ', '4000  ', '6000  ', '8000  ', '10000  ','12000 s'])



# Bolding horizontal line at y = 0

vendor_graph.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .70)



# Remove the label of the x-axis

vendor_graph.xaxis.label.set_visible(False)

vendor_graph.yaxis.label.set_visible(False)



# Add signature bar

vendor_graph.text(x = -.66, # Adjusts left side of signature bar

               y = -2500,  

               s = '   ©KAGGLE                                                 Source: NYC Taxi and Limousine Commission (TLC)   ', # copyright symbol ALT + 0169

              fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey') 



# Adding a title and a subtitle

vendor_graph.text(x =-.66, y = 13800, s = 'Store-and-forward trips found for Vendor 1 only',

               fontsize =20 , weight = 'bold', alpha = .90)

vendor_graph.text(x = -.66, y = 13000.3, 

               s = 'However, server connection does not have much bearing on the high number of outliers',

              fontsize = 14, alpha = .85)

# Format legend

vendor_graph.legend(title = 'store_and_fwd_flag', bbox_to_anchor = (.80,1),loc = 2, fontsize=12)

plt.show()



# Statistical summary

train_data.groupby(['vendor_id','store_and_fwd_flag'])['store_and_fwd_flag'].count().unstack().fillna(0)

# Settings

import matplotlib

matplotlib.style.use('fivethirtyeight')



# Create boxplot

plt.figure(figsize=(8.5,5))

passenger_graph = sns.boxplot(x = 'passenger_count', y = 'trip_duration', data = train_data, 

                          palette = 'gist_rainbow', linewidth = 2.3)



# Customize tick size

passenger_graph.tick_params(axis = 'both', which = 'major', labelsize = 12)



# Customize tick labels of the y-axis

passenger_graph.set_yticklabels(labels = [-10, '0  ', '2000  ', '4000  ', '6000  ', '8000  ', '10000  ','12000 s'])



# Bolding horizontal line at y = 0

passenger_graph.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .70)



# Add an extra vertical line by tweaking the range of the x-axis

#month_graph.set_xlim(left = -1, right = 6)



# Remove the label of the x-axis

passenger_graph.xaxis.label.set_visible(False)

passenger_graph.yaxis.label.set_visible(False)



# Add signature bar

passenger_graph.text(x = -1.1, # Adjusts left side of signature bar

               y = -2500,  

               s = '   ©KAGGLE                                                 Source: NYC Taxi and Limousine Commission (TLC)   ', # copyright symbol ALT + 0169

              fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey') 



# Alternative signature bar

# fte_graph.text(x = 1967.1, y = -6.5,

#               s = '________________________________________________________________________________________________________________',

#               color = 'grey', alpha = .70)

# fte_graph.text(x = 1966.1, y = -9,

#               s ='   ©DATAQUEST                                                                               Source: National Center for Education Statistics   ', # copyright symbol ALT + 0169

#               fontsize = 14, color = 'grey', alpha = .7)



# # Adding a title and a subtitle

passenger_graph.text(x =-1.05, y = 13800, s = "Passenger count does not have much effect on trip duration",

               fontsize =20 , weight = 'bold', alpha = .90)

passenger_graph.text(x = -1.05, y = 13000.3, 

               s = 'Median trip times remain similar despite more passengers being aboard',

              fontsize = 14, alpha = .85)

plt.show()



# Statistical summary

train_data.groupby('passenger_count')['trip_duration'].describe().transpose()
# Trips by Hour and Day of Week

trip_duration_median = train_data['trip_duration'].median()

plt.figure(figsize=(8.5,5))

pickup_hourday = train_data.groupby(['pickup_hour','pickup_weekday'])['trip_duration'].median().unstack()

hourday_graph = sns.heatmap(pickup_hourday[pickup_hourday>trip_duration_median],

                                   lw = .5, annot = True, cmap = 'GnBu', fmt = 'g',annot_kws = {"size":10} )

# Customize tick label size

hourday_graph.tick_params(axis = 'both', which = 'major', labelsize = 10)



# Customize tick labels of the y-axis

hourday_graph.set_xticklabels(labels = ['Mon', 'Tue', 'Wed','Thu','Fri','Sat','Sun'])



# Bolding horizontal line at y = 0

hourday_graph.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .70)



# Remove the label of the x-axis

hourday_graph.xaxis.label.set_visible(False)



# Add signature bar

hourday_graph.text(x = -.8,  y = -4,

                   s = ' ©KAGGLE                                          Source: NYC Taxi and Limousine Commission (TLC)   ',

fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey') 



# # Adding a title and a subtitle

hourday_graph.text(x =-.8, y = 27, s = "Trip durations vary greatly depending on day of week",

               fontsize =20 , weight = 'bold', alpha = .90)

hourday_graph.text(x =-.8, y = 25.5, 

               s = 'Median trip times longest during office hours and weekend nights',

              fontsize = 14, alpha = .85)



# plt.ylabel('pickup_hour (military time)')

# plt.xlabel('pickup_weekday (Mon - Sun)')

# plt.title('Median Trip Duration by Pickup Hour and Day of Week')

plt.show()
# Box plot of pickups by month

import matplotlib

matplotlib.style.use('fivethirtyeight')



# Create boxplot

plt.figure(figsize=(8.5,5))

month_graph = sns.boxplot(x = 'pickup_month', y = 'trip_duration', data = train_data, 

                          palette = 'gist_rainbow', linewidth = 2.3)



# Customize tick size

month_graph.tick_params(axis = 'both', which = 'major', labelsize = 12)



# Customize tick labels of the y-axis

month_graph.set_yticklabels(labels = [-10, '0  ', '2000  ', '4000  ', '6000  ', '8000  ', '10000  ','12000 s'])



# Bolding horizontal line at y = 0

month_graph.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .70)





# Add an extra vertical line by tweaking the range of the x-axis

#month_graph.set_xlim(left = -1, right = 6)



# Remove the label of the x-axis

month_graph.xaxis.label.set_visible(False)

month_graph.yaxis.label.set_visible(False)



# Add signature bar

month_graph.text(x = -1.1, # Adjusts left side of signature bar

               y = -2500,  

               s = '   ©KAGGLE                                                 Source: NYC Taxi and Limousine Commission (TLC)   ', # copyright symbol ALT + 0169

              fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey') 



# Alternative signature bar

# fte_graph.text(x = 1967.1, y = -6.5,

#               s = '________________________________________________________________________________________________________________',

#               color = 'grey', alpha = .70)

# fte_graph.text(x = 1966.1, y = -9,

#               s ='   ©DATAQUEST                                                                               Source: National Center for Education Statistics   ', # copyright symbol ALT + 0169

#               fontsize = 14, color = 'grey', alpha = .7)



# # Adding a title and a subtitle

month_graph.text(x =-1.05, y = 13800, s = "Month of transaction has minimal effect on trip duration",

               fontsize =20 , weight = 'bold', alpha = .90)

month_graph.text(x = -1.05, y = 13000.3, 

               s = 'Median trip times hover around ~650 seconds throughout the year',

              fontsize = 14, alpha = .85)

plt.show()



# Statistical summary

train_data.groupby('pickup_month')['trip_duration'].describe().transpose()
longitude = list(train_data.pickup_longitude) + list(train_data.dropoff_longitude)

latitude = list(train_data.pickup_latitude) + list(train_data.dropoff_latitude)

plt.figure(figsize = (10,8))

plt.plot(longitude,latitude,'.',alpha = .40, markersize = .8)

plt.title('Trip Plots')

plt.show()
# Create data frame of coordinates

loc_df = pd.DataFrame()

loc_df['longitude'] = longitude

loc_df['latitude'] = latitude



# Clusters of New York

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=15, random_state=2, n_init = 10).fit(loc_df)

loc_df['label'] = kmeans.labels_

loc_df = loc_df.sample(200000)

plt.figure(figsize = (12,7))

for label in loc_df.label.unique():

    plt.plot(loc_df.longitude[loc_df.label == label],loc_df.latitude[loc_df.label == label],'.', alpha = 0.8, markersize = 0.8)

plt.title('Clusters of New York')

plt.show()
# Correlations to trip_duration

corr = train_data.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

cor_dict = corr['trip_duration'].to_dict()

del cor_dict['trip_duration']

print("List the numerical features in decending order by their correlation with trip_duration:\n")

for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):

    print("{0}: {1}".format(*ele))

    

# Correlation matrix heatmap

corrmat = train_data.corr()

plt.figure(figsize=(12, 7))



# Number of variables for heatmap

k = 76

cols = corrmat.nlargest(k, 'trip_duration')['trip_duration'].index

cm = np.corrcoef(train_data[cols].values.T)



# Generate mask for upper triangle

mask = np.zeros_like(cm, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.set(font_scale=1)

sns.heatmap(cm, mask=mask, cbar=True, annot=True, square=True,\

                 fmt='.2f',annot_kws={'size': 12}, yticklabels=cols.values,\

                 xticklabels=cols.values, cmap = 'coolwarm',lw = .1)

plt.show() 
# Check for categorical variables

train_data.head()
# Encode categorical variables

train_data['store_and_fwd_flag'] = train_data['store_and_fwd_flag'].map({'N':0,'Y':1})
# Remove unnecessary features

train_data.drop(['pickup_date','pickup_time','dropoff_date', 'dropoff_time','id'], 

                axis = 1, inplace = True)
train_data.columns
# Split

# Create matrix of features

X = train_data[['vendor_id', 'passenger_count', 'pickup_longitude',

       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',

       'store_and_fwd_flag','pickup_month', 'pickup_hour',

       'pickup_weekday']] # double brackets!



# Create array of target variable 

y = train_data['trip_duration']



# Create train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#  Import model

from sklearn.linear_model import LinearRegression



#  Instantiate model object

lreg = LinearRegression()



# Fit to training data

lreg.fit(X_train,y_train)

print(lreg)



# Predict

y_pred_lreg = lreg.predict(X_test)



# Score It

from sklearn import metrics

print('\nLinear Regression Performance Metrics')

print('R^2=',metrics.explained_variance_score(y_test,y_pred_lreg))

print('MAE:',metrics.mean_absolute_error(y_test,y_pred_lreg))

print('MSE:',metrics.mean_squared_error(y_test,y_pred_lreg))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred_lreg)))
# Fit

# Import model

from sklearn.tree import DecisionTreeRegressor



# Instantiate model object

dtree = DecisionTreeRegressor()



# Fit to training data

dtree.fit(X_train,y_train)

print(dtree)



# Predict

y_pred_dtree = dtree.predict(X_test)



# Score It

from sklearn import metrics

print('\nDecision Tree Regression Performance Metrics')

print('R^2=',metrics.explained_variance_score(y_test,y_pred_dtree))

print('MAE:',metrics.mean_absolute_error(y_test,y_pred_dtree))

print('MSE:',metrics.mean_squared_error(y_test,y_pred_dtree))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred_dtree)))
# Fit 

# Import model

from sklearn.ensemble import RandomForestRegressor 



# Instantiate model object

rforest = RandomForestRegressor(n_estimators = 20, n_jobs = -1)



# Fit to training data

rforest = rforest.fit(X_train,y_train)

print(rforest)



# Predict

y_pred_rforest = rforest.predict(X_test)



# Score It

from sklearn import metrics

print('\nRandom Forest Regression Performance Metrics')

print('R^2 =',metrics.explained_variance_score(y_test,y_pred_rforest))

print('MAE',metrics.mean_absolute_error(y_test, y_pred_rforest))

print('MSE',metrics.mean_squared_error(y_test, y_pred_rforest))

print('RMSE',np.sqrt(metrics.mean_squared_error(y_test, y_pred_rforest)))
# Load test data

test_data = pd.read_csv('../input/test.csv')



# Test data info

test_data.info()



# Test data shape

print('shape',test_data.shape)
# Convert timestamps to date objects

test_data['pickup_datetime'] = pd.to_datetime(test_data.pickup_datetime) # Pickups



# Delimit pickup_datetime variable 

test_data['pickup_date'] = test_data['pickup_datetime'].dt.date # Extract date

test_data['pickup_time'] = test_data['pickup_datetime'].dt.time # Extract time



# Additional pickup features

test_data['pickup_month'] = test_data['pickup_datetime'].dt.month # Extract month



#train_data['pickup_YYYYMM'] = train_data['pickup_datetime'].apply(lambda x: x.strftime('%Y%m')) # Extract yearmonth

test_data['pickup_hour'] = test_data['pickup_datetime'].dt.hour # Extract hour

test_data['pickup_weekday'] = test_data['pickup_datetime'].dt.dayofweek # Extract day of week



# Encode categorical variables

test_data['store_and_fwd_flag'] = test_data['store_and_fwd_flag'].map({'N':0,'Y':1})

# Create new matrix of features from test data

X_test= test_data[['vendor_id', 'passenger_count', 'pickup_longitude',

       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',

       'store_and_fwd_flag','pickup_month', 'pickup_hour',

       'pickup_weekday']]



# Feed features into random forest

y_pred= rforest.predict(X_test)
# Create contest submission

submission = pd.DataFrame({

    'Id':test_data['id'],

    'trip_duration': y_pred

})

submission.to_csv('mytaxisubmission.csv',index = False)
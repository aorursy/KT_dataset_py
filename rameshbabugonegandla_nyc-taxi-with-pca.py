# Loading Libraries
import pandas as pd # for data analysis

import numpy as np # for scientific calculation

import seaborn as sns # for statistical plotting

import datetime # for working with date fields

import matplotlib.pyplot as plt # for plotting

%matplotlib inline

import math # for mathematical calculation
import os

#Reading NYC Taxi Trip given Data Set.

import os

for dirname, _, filenames in os.walk('kaggle/input/NYC_taxi_trip_train.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Reading NYC taxi trip given Data Set.

nyc_taxi=pd.read_csv('/kaggle/input/NYC_taxi_trip_train.csv') 
#Perform Pandas profiling to understand quick overview of columns



#import pandas_profiling

#report = pandas_profiling.ProfileReport(nyc_taxi)

#covert profile report as html file

#report.to_file("nyc_taxi.html")
# Checking Null Values : We can See there are No Null Values 

nyc_taxi.isnull().sum()
#Checking shape of data

#Observation: It contains 1.4 million records approx. and 11 columns (10 features with 1 feature as a target variable)

nyc_taxi.shape
#Checking duplicates in the given dataset.

#Observations: No duplicates exists as it's row count shows '0'.

check_duplicates = nyc_taxi[nyc_taxi.duplicated()]

print(check_duplicates.shape)
#Exploring data by using info() method. It doesn't contains any null values.

#Observation: No null values exists.

nyc_taxi.info()
# Verifying top 2 sample records of data.

# Observation: The data consists of, vendor_id, pickup and dropoff datetime, longitude and latitude information, trip_duration

# values plays major part in predicting the tripduration here.

nyc_taxi.head(2)
# Describe method is used to view some basic statistical details like percentile, mean, std etc. of a data frame of numeric values.

#Observation: Due to huge dataset and the columns values has been given in the form of +/- (e.g., longitude and Latitude)

# it shows data in the form of exponentials.Moving ahead with EDA and visualization to understand data better.

nyc_taxi.describe()
# Distance  function to calculate distance between given longitude and latitude points.

# Observation: This piece of code taken from blogs. When I thought how to get pickup point and drop point information

# I found this code and I can able to calculate distance here. It's as been called as 'Haversine Formula'

from math import radians, cos, sin, asin, sqrt



def distance(lon1, lat1, lon2, lat2):

    """

    Calculate the great circle distance between two points 

    on the earth (specified in decimal degrees)

    """

    # convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])



    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    c = 2 * asin(sqrt(a)) 

    r = 6371 # Radius of earth in kilometers. Use 3956 for miles

    return c * r
#Calculate Trip Distance & Speed here.

#Observation: Introduced New Columns Distance and Speed here.

# Converted dropoff_datetime and pickup_datetime into datetime format datatype

nyc_taxi['distance'] = nyc_taxi.apply(lambda x: distance(x['pickup_longitude'],x['pickup_latitude'],x['dropoff_longitude'],x['dropoff_latitude']), axis = 1)

nyc_taxi['speed']= (nyc_taxi.distance/(nyc_taxi.trip_duration/3600))

nyc_taxi['dropoff_datetime']= pd.to_datetime(nyc_taxi['dropoff_datetime']) 

nyc_taxi['pickup_datetime']= pd.to_datetime(nyc_taxi['pickup_datetime'])
#Verify the column list

nyc_taxi.columns

#Copied dataframe into another dataframe.

# Observation: Using another dataframe for data visualization and keeping original copy for ML pipeline.

nyc_taxi_visual = nyc_taxi.copy()
#Verifying columns.

nyc_taxi_visual.columns
#Drop unused columns for data visualization

nyc_taxi_visual = nyc_taxi_visual.drop(['pickup_datetime','dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','store_and_fwd_flag'],axis=1)

nyc_taxi_visual.columns
# Verifying datatype, count, null values by using info() method.

nyc_taxi_visual.info()
#Converting distance and speed values into int datatype.

nyc_taxi_visual['distance']=nyc_taxi_visual['distance'].apply(lambda x:int(x))

nyc_taxi_visual['speed']=nyc_taxi_visual['speed'].apply(lambda x:int(x))
#Verifying the datatype for all columns

nyc_taxi_visual.info()
# Seaborn scatter plot with regression line

# Observation: We can see her we are having outliers when distance is >50 miles  and trip duration is >15000 seconds and

# we can say most of the trip_duration >15000 seconds mostly related to long distance or due to traffic jam on odd days.

# We can see regression line fits here when we remove outliers.

sns.lmplot(x='trip_duration', y='distance', data=nyc_taxi_visual, aspect=2.0, scatter_kws={'alpha':0.8})
# Removing outliers for distance and trip_duration.

nyc_taxi_visual_final=nyc_taxi_visual[nyc_taxi_visual['distance']<=600]

nyc_taxi_visual_final=nyc_taxi_visual[nyc_taxi_visual['trip_duration']<=36000]

nyc_taxi_visual_final.head(1)
# Distribution plot for trip_Duration.

# Observation: Data is right skewed here and most of datapoints is having very short trip_durations.

# Will apply scaling techniques before we train the model.

sns.distplot(nyc_taxi_visual_final['trip_duration'],kde = False)
# Distribution plot for Passenger Count.

# Observations: Most of the times, only single passenger has booked taxi. New york city most of the times, only one passenger

# travels due to population density and business center. It makes sense here. Very few trips for > 3 passengers.

sns.distplot(nyc_taxi_visual_final['passenger_count'],kde=False, bins=None)

plt.title('Distribution of Passenger Count')

plt.show()
#Verifying column details

nyc_taxi_visual_final.columns
# Groupby function to calculate passenger_count who has taken trips from Vendors.

#Observations: Vendor_id 1 is having more trips for passenger 1 and 

# for Vendor_id 2 is having good number trips for passenger 1 and when passenger_count> 3 when compare to vendor_id 1

nyc_taxi_visual_final.groupby(by=['vendor_id','passenger_count'])['passenger_count'].count()
#Box plot for passenger_count for both vendors.

# Observation: Based on the given huge dataset it's clear that, we are having outliers for both vendors when passenger

# count increases more than 2.

# for vendor_id 1 we can see outliers when passenger_count is 0. might be because of empty trips or some other reasons.

plt.figure(figsize=(15,5))

sns.boxplot(x="vendor_id", y="passenger_count",data = nyc_taxi_visual_final)
#Box plot for trip_duration for both vendors.

#Observation: We can see more outliders for both vendors when trip_duration is > 1000 seconds. It's a fact that

# new_york is one of the costliest and expensive life style city and most of the passenger can book trips <10 minutes travel.

# Might be these outliers trip_duration belongs to tourists.

plt.figure(figsize=(15,5))

sns.boxplot(x="vendor_id", y="trip_duration",data = nyc_taxi_visual_final)
#np.max(nyc_taxi_visual_final['distance'])
# Plotly Scatter bubble chart used to visualize trip_duration and distance details vendor_id wise distribution.

# Observation: Most of datapoints lies between distance<50 and trip_duration <15K

# Vendor_id 1 is having outliers for distance > 100 miles.

# Vendor_id 2 is having outliers for trip_duration > 15000 seconds.

import plotly.express as px

fig=px.scatter(nyc_taxi_visual_final,

                           x='trip_duration',

                           y='distance',

                           size='trip_duration',color='vendor_id'

                           )

fig.update_layout(title="Trip Duration details vendor_id wise distribution")

fig.show()
# Plotly Pie chart used to visualize Share of each vendor_id in the given data set.

# Observation: Vendor_id 1 is having 46.5% and 2 is having 53.5% share in NYC Taxi Trips.

# Vendor_id 2 is having more than 15% of share when we compare with Vendor_id 1 share contribution.

import plotly.graph_objects as go

df1=nyc_taxi_visual['vendor_id'].value_counts().reset_index()

fig=go.Figure(data=[go.Pie(labels=df1['index'],

                          values=df1['vendor_id'],

                          hole=.4,

                          title="Share of each Vendor")])

fig.update_layout(title="NYC_Taxi Vendor Details")

fig.show()
# Plotly BAr chart used to visualize number of trips contributed by each vendor in the given data set.

# Observation: Vendor_id 2 is having more number of trips when compare to vendor_id 1.

sns.barplot(nyc_taxi_visual_final['vendor_id'].value_counts().index, nyc_taxi_visual_final['vendor_id'].value_counts().values, alpha=0.8, palette = sns.color_palette('RdBu'))
#Analyzing given datapoint based on distance travelled by passenger.  

#Need to remove

nyc_taxi.sort_values(by='distance',ascending=False).head(10)
# Dropping unused columns

nyc_taxi_final=nyc_taxi.drop(['vendor_id','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','store_and_fwd_flag'],axis=1)

# Creating new feature columns.

nyc_taxi_final['pickup_min'] = nyc_taxi_final['pickup_datetime'].apply(lambda x : x.minute)

nyc_taxi_final['pickup_hour'] = nyc_taxi_final['pickup_datetime'].apply(lambda x : x.hour)

nyc_taxi_final['pickup_day'] = nyc_taxi_final['pickup_datetime'].apply(lambda x : x.day)

nyc_taxi_final['pickup_month']= nyc_taxi_final['pickup_datetime'].apply(lambda x : int(x.month))

nyc_taxi_final['pickup_weekday'] = nyc_taxi_final['pickup_datetime'].dt.day_name()

nyc_taxi_final['pickup_month_name'] = nyc_taxi_final['pickup_datetime'].dt.month_name()



nyc_taxi_final['drop_hour'] = nyc_taxi_final['dropoff_datetime'].apply(lambda x : x.hour)

nyc_taxi_final['drop_month'] = nyc_taxi_final['dropoff_datetime'].apply(lambda x : int(x.month))

nyc_taxi_final['drop_day'] = nyc_taxi_final['dropoff_datetime'].apply(lambda x : x.day)

nyc_taxi_final['drop_min'] = nyc_taxi_final['dropoff_datetime'].apply(lambda x : x.minute)



#Verifying newly created columns.

nyc_taxi_final.columns
## Removing all those records where speed is less than 1 and distance is 0

print(nyc_taxi_final.shape)

df=nyc_taxi_final[(nyc_taxi_final['speed']<1)&(nyc_taxi_final['distance']==0)]

nyc_taxi_final.drop(df.index,inplace=True)

print(nyc_taxi_final.shape)
# Identified some of trips are not valid from the given dataset.

# For e.g., Index 531 clearly says that pick up adn drop off date and time is having morethan 23 hours trip_duration

# by covering distance only 3 miles which is not possible in realtime scenarios. Removing those outliers. Total 1416 records.

nyc_taxi_final[(nyc_taxi_final['pickup_day']< nyc_taxi_final['drop_day'])& (nyc_taxi_final['trip_duration']> 10000) &(nyc_taxi_final['distance'] <5) & (nyc_taxi_final['pickup_hour']<23)]
# Dropping records for those whose pickup and drop timings are more and distance travel <3 miles. (Outliers.)

print(nyc_taxi_final.shape)

df=nyc_taxi_final[(nyc_taxi_final['pickup_day']< nyc_taxi_final['drop_day'])& (nyc_taxi_final['trip_duration']> 10000) &(nyc_taxi_final['distance'] <5) & (nyc_taxi_final['pickup_hour']<23)]

nyc_taxi_final.drop(df.index,inplace=True)

print(nyc_taxi_final.shape)
# Droppring records where speed and distance is <1. (Outliers)

print(nyc_taxi_final.shape)

df=nyc_taxi_final[(nyc_taxi_final['speed']<1) & (nyc_taxi_final['distance']< 1) ]

nyc_taxi_final.drop(df.index,inplace=True)

print(nyc_taxi_final.shape)
# Removing outliers identified based on trip_duration and distance.

nyc_taxi_final[nyc_taxi_final['trip_duration']/60 >10000][['trip_duration','distance']]

print(nyc_taxi_final.shape)

nyc_taxi_final[nyc_taxi_final['trip_duration']/60 >10000]['trip_duration']

nyc_taxi_final.drop([978383,680594,355003],inplace=True)

print(nyc_taxi_final.shape)
# Removing outliers whose distance is less 200 meters. In real scenario, no-one will pick taxi for lesstance 200 meters. 

print(nyc_taxi_final.shape)

df=nyc_taxi_final[nyc_taxi_final['distance']< .2]

nyc_taxi_final.drop(df.index,inplace=True)

print(nyc_taxi_final.shape)
# Removing outliers those trips where passenger_count is 0.

print(nyc_taxi_final.shape)

df=nyc_taxi_final[nyc_taxi_final['passenger_count']==0]

nyc_taxi_final.drop(df.index,inplace=True)

print(nyc_taxi_final.shape)
# Verifying whether given data set is having other than 2016 year or not. 

# Observation: It contains only 2016 year data.

import datetime as dt

print(nyc_taxi_final[nyc_taxi_final['dropoff_datetime'].dt.year>2016])

print(nyc_taxi_final[nyc_taxi_final['dropoff_datetime'].dt.year<2016])
# Removing outliers where trip_duration <120 seconds. In real-time scenario passengers will take trip for more than 2 mins.

print(nyc_taxi_final.shape)

df=nyc_taxi_final[nyc_taxi_final['trip_duration']<120]

nyc_taxi_final.drop(df.index,inplace=True)

print(nyc_taxi_final.shape)
# Distribution plot to verify the speed of trip.

# Observations: Most of trips is having speed < 40 miles/per hour. It's valid in newyork city trips.

dist_plot=nyc_taxi_final[nyc_taxi_final['speed']<100]['speed']

sns.distplot(dist_plot,bins=10)
# Distribution plot to verify the speed of trip for complete dataset.

# Observations: Most of the trips is having speed <50 miles/per hour and removed outliers.

print(nyc_taxi_final.shape)

df=nyc_taxi_final[nyc_taxi_final['speed']>50]['speed']

sns.distplot(df,bins=10)

nyc_taxi_final.drop(df.index,inplace=True)

print(nyc_taxi_final.shape)
#Verify column details.

nyc_taxi_final.columns
# Verifying the Day-Wise trip counts.

#Observation: We are having less trips on sunday and monday here.

print("Day-wise pickup totals")

print(nyc_taxi_final['pickup_weekday'].value_counts())

# Countplot visualization for Day-wise trip counts.

# Observations: Friday and Saturday is having more trips when compare to other days.

sns.countplot(x='pickup_weekday',data=nyc_taxi_final)
# Histogram plot to visualize for hour-wise trips 

# Observations: Most ot trip counts is having between 5am to 23 pm and 0am(midnight) to 2am.



figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))



nyc_taxi_final['pickup_hour']=nyc_taxi_final['pickup_datetime'].dt.hour

nyc_taxi_final.pickup_hour.hist(bins=24,ax=ax[0])

ax[0].set_title('Distribution of pickup hours')



nyc_taxi_final['dropoff_hour']=nyc_taxi_final['dropoff_datetime'].dt.hour

nyc_taxi_final.dropoff_hour.hist(bins=24,ax=ax[1])

ax[1].set_title('Distribution of dropoff hours')
#Import Sklearn and models

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score, KFold

#Verifying final dataframe columns

nyc_taxi_final.columns
# Due to huge dataset, training the model by applying datasampling technique. Random Sample taken 500000 records.

nyc_taxi_final_sampling=nyc_taxi_final.sample(n=500000,replace="False")
#Verify the shape of data

nyc_taxi_final_sampling.shape
# Dropping unused columns and copied required columns to the feature_columns to train the model.

# Used to verify co-efficient values.

feature_columns=nyc_taxi_final_sampling.drop(['id','pickup_month_name','pickup_weekday','pickup_datetime','dropoff_datetime','trip_duration','passenger_count','speed'],axis=1)
#Verifying whether feature_columns is having null values or not.

nyc_taxi_final_sampling.distance=nyc_taxi_final_sampling.distance.astype(np.int64)

nyc_taxi_final_sampling.info()
#Applying Standard Scaler

X2=nyc_taxi_final_sampling.drop(['id','pickup_month_name','pickup_weekday','pickup_datetime','dropoff_datetime','trip_duration','passenger_count','speed'],axis=1)

X1=preprocessing.scale(X2)

X=pd.DataFrame(X1)

y=nyc_taxi_final_sampling['trip_duration']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)

reg =linear_model.LinearRegression()



reg.fit(X_train,y_train)

print("reg.intercept_=> %10.10f" %(reg.intercept_))

print(list(zip(feature_columns, reg.coef_)))

y_pred=reg.predict(X_test)

rmse_val=np.sqrt(metrics.mean_squared_error(y_test, y_pred))



# Null RMSE

y_null = np.zeros_like(y_test, dtype=int)

y_null.fill(y_test.mean())

N_RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_null))

# Metrics

print('Mean Absolute Error    :', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error     :', metrics.mean_squared_error(y_test, y_pred)) 

print("Root Mean Squared Error = ",rmse_val)

print("Null RMSE = ",N_RMSE)

if N_RMSE < rmse_val:print("Model is Not Doing Well Null RMSE Should be Greater")

else:print("Model is Doing Well Null RMSE is Greater than RMSE")

# Train RMSE

y_pred_test=reg.predict(X_train)

rmse_val=np.sqrt(metrics.mean_squared_error(y_train, y_pred_test))

print('Train Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_test)))

# Error Percentage

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

print("Maximum Error is :",df.Error.max())

print("Minimum Error is :",df.Error.min())

# Score

scores = cross_val_score(reg,X_train,y_train,cv=5)

print("Mean cross-validation score: %.2f" % scores.mean())



import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error



from sklearn.model_selection import RandomizedSearchCV ,cross_val_score, KFold



X2=nyc_taxi_final_sampling.drop(['id','pickup_month_name','pickup_weekday','pickup_datetime','dropoff_datetime','trip_duration','passenger_count','speed'],axis=1)

X1=preprocessing.scale(X2)

X=pd.DataFrame(X1)

y=nyc_taxi_final_sampling['trip_duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)



model = xgb.XGBRegressor()

model.fit(X_train,y_train)

print(model)

y_pred = model.predict(data=X_test)



rmse_val=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# Null RMSE

y_null = np.zeros_like(y_test, dtype=int)

y_null.fill(y_test.mean())

N_RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_null))

# Metrics

print('Mean Absolute Error    :', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error     :', metrics.mean_squared_error(y_test, y_pred)) 

print("Root Mean Squared Error = ",rmse_val)

print("Null RMSE = ",N_RMSE)

if N_RMSE < rmse_val:print("Model is Not Doing Well Null RMSE Should be Greater")

else:print("Model is Doing Well Null RMSE is Greater than RMSE")

# Train RMSE

y_pred_test=model.predict(X_train)

rmse_val=np.sqrt(metrics.mean_squared_error(y_train, y_pred_test))

print('Train Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_test)))

# Error Percentage

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

print("Maximum Error is :",df.Error.max())

print("Minimum Error is :",df.Error.min())

# Score

scores = cross_val_score(model,X_train,y_train,cv=5)

print("Mean cross-validation score: %.2f" % scores.mean())

from sklearn.linear_model import Ridge



X2=nyc_taxi_final_sampling.drop(['id','pickup_month_name','pickup_weekday','pickup_datetime','dropoff_datetime','trip_duration','passenger_count','speed'],axis=1)

X1=preprocessing.scale(X2)

X=pd.DataFrame(X1)

y=nyc_taxi_final_sampling['trip_duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)



ridgeReg = Ridge(alpha=0.05, normalize=True)

ridgeReg.fit(X_train,y_train)



y_pred = ridgeReg.predict(X_test)



rmse_val=np.sqrt(metrics.mean_squared_error(y_test, y_pred))



# NULL RMSE

y_null = np.zeros_like(y_test, dtype=int)

y_null.fill(y_test.mean())

N_RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_null))

# Metrics

print('Mean Absolute Error    :', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error     :', metrics.mean_squared_error(y_test, y_pred)) 

print("Root Mean Squared Error = ",rmse_val)

print("Null RMSE = ",N_RMSE)

if N_RMSE < rmse_val:print("Model is Not Doing Well Null RMSE Should be Greater")

else:print("Model is Doing Well Null RMSE is Greater than RMSE")

# Train RMSE

y_pred_test=ridgeReg.predict(X_train)

rmse_val=np.sqrt(metrics.mean_squared_error(y_train, y_pred_test))

print('Train Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_test)))

# Error Percentage

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

print("Maximum Error is :",df.Error.max())

print("Minimum Error is :",df.Error.min())

# Score

scores = cross_val_score(ridgeReg,X_train,y_train,cv=5)

print("Mean cross-validation score: %.2f" % scores.mean())
from sklearn.linear_model import RidgeCV

## training the model



X2=nyc_taxi_final_sampling.drop(['id','pickup_month_name','pickup_weekday','pickup_datetime','dropoff_datetime','trip_duration','passenger_count','speed'],axis=1)

X1=preprocessing.scale(X2)

X=pd.DataFrame(X1)

y=nyc_taxi_final_sampling['trip_duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)



ridgeRegCV = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])

ridgeRegCV.fit(X_train,y_train)





y_pred = ridgeRegCV.predict(X_test)



rmse_val=np.sqrt(metrics.mean_squared_error(y_test, y_pred))



# Null RMSE

y_null = np.zeros_like(y_test, dtype=int)

y_null.fill(y_test.mean())

N_RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_null))

# Metrics

print('Mean Absolute Error    :', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error     :', metrics.mean_squared_error(y_test, y_pred)) 

print("Root Mean Squared Error = ",rmse_val)

print("Null RMSE = ",N_RMSE)

if N_RMSE < rmse_val:print("Model is Not Doing Well Null RMSE Should be Greater")

else:print("Model is Doing Well Null RMSE is Greater than RMSE")

# Train RMSE

y_pred_test=ridgeRegCV.predict(X_train)

rmse_val=np.sqrt(metrics.mean_squared_error(y_train, y_pred_test))

print('Train Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_test)))

# Error Percentage

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

print("Maximum Error is :",df.Error.max())

print("Minimum Error is :",df.Error.min())

# Score

scores = cross_val_score(ridgeRegCV,X_train,y_train,cv=5)

print("Mean cross-validation score: %.2f" % scores.mean())
#Metrics Overview

df = {'Model_Before_PCA':['Linear_Reg', 'XGB', 'Ridge','RidgeCV'],

'RMSE': ['875', '1249', '1605','876' ],

'NULL_RMSE': ['1701','1701','1701','1701'],

'Max_error': ['66971','84155','85443','67179'],

'Min_error': ['-80844','-52886','-4015','-78683'],

'Score':['72','44','11','72']}

print("Metrics Overview Before_PCA")

dataframe = pd.DataFrame(df, columns=['Model','RMSE', 'NULL_RMSE', 'Max_error', 'Min_error','Score'])

dataframe
#Verify the shape of data

nyc_taxi_final_sampling.shape
#Aligning data for PCA.

nyc_taxi_final_sampling.columns

nyc_taxi_pca=nyc_taxi_final_sampling.copy()

nyc_taxi_pca.drop(['id','pickup_weekday', 'pickup_month_name','pickup_datetime','dropoff_datetime','speed'],axis=1,inplace=True)
# seperate target variable for PCA.

target = nyc_taxi_pca['trip_duration']
from sklearn import datasets

from sklearn.decomposition import PCA
# PCA

# normalize data

nyc_taxi_pca_norm = (nyc_taxi_pca - nyc_taxi_pca.mean()) / nyc_taxi_pca.std()



pca = PCA(n_components=12) # 12 features

pca.fit_transform(nyc_taxi_pca_norm.values)

print (pca.explained_variance_ratio_)

#print (nyc_taxi_final.feature_names)

print (pca.explained_variance_)

variance_ratio_cum_sum=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(variance_ratio_cum_sum)

print (pca.components_)
# Taken variance ratio of 7 PCA components at 93.6%.

pca.explained_variance_ratio_[:7].sum()
#Plot Elbow Curve

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')

plt.annotate('7',xy=(7, .93))
# consider first 7 components as they are explaining the 93% of variation in the data

x_pca = PCA(n_components=7)

nyc_taxi_pca_norm_final = x_pca.fit_transform(nyc_taxi_pca_norm)
# correlation between the variables after transforming the data with PCA is 0

correlation = pd.DataFrame(nyc_taxi_pca_norm_final).corr()

sns.heatmap(correlation, vmax=1, square=True,cmap='viridis')

plt.title('Correlation between different features')
X2=preprocessing.scale(nyc_taxi_pca_norm_final)

X=pd.DataFrame(X2)

y=target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)

reg =linear_model.LinearRegression()



reg.fit(X_train,y_train)

print("reg.intercept_=> %10.10f" %(reg.intercept_))

print(list(zip(feature_columns, reg.coef_)))

y_pred=reg.predict(X_test)

rmse_val=np.sqrt(metrics.mean_squared_error(y_test, y_pred))



# NULL RMSE

y_null = np.zeros_like(y_test, dtype=int)

y_null.fill(y_test.mean())

N_RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_null))

# Metrics

print('Mean Absolute Error    :', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error     :', metrics.mean_squared_error(y_test, y_pred)) 

print("Root Mean Squared Error = ",rmse_val)

print("Null RMSE = ",N_RMSE)

if N_RMSE < rmse_val:print("Model is Not Doing Well Null RMSE Should be Greater")

else:print("Model is Doing Well Null RMSE is Greater than RMSE")

# Train RMSE

y_pred_test=reg.predict(X_train)

rmse_val=np.sqrt(metrics.mean_squared_error(y_train, y_pred_test))

print('Train Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_test)))

# Error Percentage

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

print("Maximum Error is :",df.Error.max())

print("Minimum Error is :",df.Error.min())

# Score

scores = cross_val_score(reg,X_train,y_train,cv=5)

print("Mean cross-validation score: %.2f" % scores.mean())
import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import RandomizedSearchCV ,cross_val_score, KFold



X2=preprocessing.scale(nyc_taxi_pca_norm_final)

X=pd.DataFrame(X2)

y=target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)



model = xgb.XGBRegressor()

model.fit(X_train,y_train)

print(model)

y_pred = model.predict(data=X_test)



rmse_val=np.sqrt(metrics.mean_squared_error(y_test, y_pred))



# Null RMSE

y_null = np.zeros_like(y_test, dtype=int)

y_null.fill(y_test.mean())

N_RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_null))

# Metrics

print('Mean Absolute Error    :', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error     :', metrics.mean_squared_error(y_test, y_pred)) 

print("Root Mean Squared Error = ",rmse_val)

print("Null RMSE = ",N_RMSE)

if N_RMSE < rmse_val:print("Model is Not Doing Well Null RMSE Should be Greater")

else:print("Model is Doing Well Null RMSE is Greater than RMSE")

    

# Train RMSE

y_pred_test=model.predict(X_train)

rmse_val=np.sqrt(metrics.mean_squared_error(y_train, y_pred_test))

print('Train Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_test)))

# Error Percentage

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

print("Maximum Error is :",df.Error.max())

print("Minimum Error is :",df.Error.min())

# Score

scores = cross_val_score(model, X_train,y_train,cv=5)

print("Mean cross-validation score: %.2f" % scores.mean())



X2=preprocessing.scale(nyc_taxi_pca_norm_final)

X=pd.DataFrame(X2)

y=target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)



ridgeReg = Ridge(alpha=0.05, normalize=True)

ridgeReg.fit(X_train,y_train)



y_pred = ridgeReg.predict(X_test)



rmse_val=np.sqrt(metrics.mean_squared_error(y_test, y_pred))



#Null RMSE

y_null = np.zeros_like(y_test, dtype=int)

y_null.fill(y_test.mean())

N_RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_null))

# Metrics

print('Mean Absolute Error    :', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error     :', metrics.mean_squared_error(y_test, y_pred)) 

print("Root Mean Squared Error = ",rmse_val)

print("Null RMSE = ",N_RMSE)

if N_RMSE < rmse_val:print("Model is Not Doing Well Null RMSE Should be Greater")

else:print("Model is Doing Well Null RMSE is Greater than RMSE")

# Train RMSE

y_pred_test=ridgeReg.predict(X_train)

rmse_val=np.sqrt(metrics.mean_squared_error(y_train, y_pred_test))

print('Train Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_test)))

# Error Percentage

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

print("Maximum Error is :",df.Error.max())

print("Minimum Error is :",df.Error.min())

# Score

scores = cross_val_score(ridgeReg, X_train,y_train,cv=5)

print("Mean cross-validation score: %.2f" % scores.mean())
X2=preprocessing.scale(nyc_taxi_pca_norm_final)

X=pd.DataFrame(X2)

y=target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)



ridgeRegCV = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])

ridgeRegCV.fit(X_train,y_train)





y_pred = ridgeRegCV.predict(X_test)



rmse_val=np.sqrt(metrics.mean_squared_error(y_test, y_pred))



# Null RMSE

y_null = np.zeros_like(y_test, dtype=int)

y_null.fill(y_test.mean())

N_RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_null))

# Metrics

print('Mean Absolute Error    :', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error     :', metrics.mean_squared_error(y_test, y_pred)) 

print("Root Mean Squared Error = ",rmse_val)

print("Null RMSE = ",N_RMSE)

if N_RMSE < rmse_val:print("Model is Not Doing Well Null RMSE Should be Greater")

else:print("Model is Doing Well Null RMSE is Greater than RMSE")

# Train RMSE

y_pred_test=ridgeRegCV.predict(X_train)

rmse_val=np.sqrt(metrics.mean_squared_error(y_train, y_pred_test))

print('Train Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_test)))

# Error Percentage

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

print("Maximum Error is :",df.Error.max())

print("Minimum Error is :",df.Error.min())

# Score

scores = cross_val_score(ridgeRegCV, X_train,y_train,cv=5)

print("Mean cross-validation score: %.2f" % scores.mean())
df = {'Model_Before_PCA':['Linear_Reg', 'XGB', 'Ridge','RidgeCV'],

'RMSE': ['875', '1249', '1605','876' ],

'NULL_RMSE': ['1701','1701','1701','1701'],

'Max_error': ['66971','84155','85443','67179'],

'Min_error': ['-80844','-52886','-4015','-78683'],

'Score':['72','44','11','72']}

print("Metrics Overview Before_PCA")

dataframe = pd.DataFrame(df, columns=['Model_Before_PCA','RMSE', 'NULL_RMSE', 'Max_error', 'Min_error','Score'])

dataframe
df = {'Model_After_PCA':['Linear_Reg', 'XGB', 'Ridge','RidgeCV'],

'RMSE': ['988', '108', '990','988' ],

'NULL_RMSE': ['1701','1701','1701','1701'],

'Max_error': ['42900','13824','44918','42900'],

'Min_error': ['-12259','-9575','-11434','-12259'],

'Score':['66','99','66','66']}

print("Metrics Overview After_PCA")

dataframe = pd.DataFrame(df, columns=['Model_After_PCA','RMSE', 'NULL_RMSE', 'Max_error', 'Min_error','Score'])

dataframe

X2=preprocessing.scale(nyc_taxi_pca_norm_final)

X=pd.DataFrame(X2)

y=target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)



model = xgb.XGBRegressor()

model.fit(X_train,y_train)

y_pred = model.predict(data=X_test)



x_ax = range(len(y_test))

plt.figure(figsize=(20,10))

plt.scatter(x_ax, y_test, s=10, color="blue", label="original")

plt.scatter(x_ax, y_pred, s=10, color="red", label="predicted")

plt.legend()

plt.show()
# From above plot you can see index:59891 is having trip_duration of 86391 acutal value(blue point) 

# predicated value (red point)

nyc_taxi_final_sampling.sort_values(by='trip_duration',ascending=False).head(2)
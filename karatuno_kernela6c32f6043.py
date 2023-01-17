#importing libraries

import numpy as np 

import pandas as pd

import os

import datetime

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeRegressor

from sklearn import tree

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

print(os.listdir("../input"))
# Load CSV

train_df = pd.read_csv("../input/train_cab.csv")
train_df.head()
train_df.dtypes
train_df['fare_amount'] = pd.to_numeric(train_df['fare_amount'], errors='coerce')

train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'], errors='coerce')
train_df.describe()
#checking null values in the data

print(train_df.isnull().sum())
#drop na

train_df = train_df.dropna(how = 'any', axis = 'rows')



#replace 0's in coordinates with null values

coord = ['pickup_longitude','pickup_latitude', 

         'dropoff_longitude', 'dropoff_latitude']



#now removing these rows with null values

for i in coord :

    train_df[i] = train_df[i].replace(0,np.nan)

    train_df    = train_df[train_df[i].notnull()]

#plotting to see the distribution of fare_amount

train_df['fare_amount'].value_counts().hist(color = 'b', edgecolor = 'k');

plt.title('Fare amount distibution'); plt.xlabel('fare_amount in $'); plt.ylabel('Count');
#checking for outliers

#but we wont use this method as it leaves out important data

q75, q25 = np.percentile(train_df.loc[:,i], [75 ,25])

iqr = q75 - q25

min = q25 - (iqr*1.5)

max = q75 + (iqr*1.5)

print(min)

print(max)
#removing the values less than 0 and greater than 99.99 percentile

train_df = train_df[ (train_df["fare_amount"] > 0 ) &

                     (train_df["fare_amount"]  <  

                      train_df["fare_amount"].quantile(.9999))]
#plotting to see the distribution of passenger_count

train_df['passenger_count'].value_counts().plot.bar(color = 'b', edgecolor = 'k');

plt.title('Passenger Counts'); plt.xlabel('Number of Passengers'); plt.ylabel('Count');
#only accepting rows with values greter thn or equal to 1 or less than 7.

train_df = train_df[(train_df["passenger_count"] >=1 ) &

                        (train_df["passenger_count"] < 7) ]
#removing the values less than 0 and greater than 99.99 percentile

coords = ['pickup_longitude','pickup_latitude', 

          'dropoff_longitude', 'dropoff_latitude']

for i in coord  : 

    train_df = train_df[(train_df[i]   > train_df[i].quantile(.001)) & 

                        (train_df[i] < train_df[i].quantile(.999))]
#adding new variables to data

def add_travel_vector_features(df):

    df['diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()

    df['diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()



add_travel_vector_features(train_df)
plot = train_df.plot.scatter('diff_longitude', 'diff_latitude')
#getting distance between pickup and dropoff location using haversine function

def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2) :

  R = 6371 #radius of earth

  dLat = deg2rad(lat2-lat1)

  dLon = deg2rad(lon2-lon1)

  a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(deg2rad(lat1)) * np.cos(deg2rad(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)

    

  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

  d = R * c

  return d



def deg2rad(deg) :

    PI=22/7



    return deg * (PI/180)

train_df['distance']= getDistanceFromLatLonInKm(train_df['pickup_longitude'],train_df['pickup_latitude'],train_df['dropoff_longitude'],train_df['dropoff_latitude'])
#checking rows with distance equal to zero

train_df[ (train_df["distance"] == 0 )]
#removing rows with distance equal to zero

train_df = train_df[ (train_df["distance"] > 0 )]
#checking co-relation between different variables for feature selevtion

cnames =  ["fare_amount","pickup_datetime","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude", "passenger_count","diff_longitude","diff_latitude", "distance"]

##Correlation analysis

#Correlation plot

df_corr = train_df.loc[:,cnames]



#Set the width and hieght of the plot

f, ax = plt.subplots(figsize=(7, 5))



#Generate correlation matrix

corr = df_corr.corr()



#Plot using seaborn library

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
#######################################   Linear Regression #############################################################

import statsmodels.api as sm

from sklearn.model_selection import train_test_split
# Bin the fare and convert to string

train_df['fare_bin'] = pd.cut(train_df['fare_amount'], bins = list(range(0, 50, 5))).astype(str)

# Uppermost bin

train_df.loc[train_df['fare_bin'] == 'nan', 'fare_bin'] = '[45+]'
#Dividing the data in 80:20 ratio wih fare_bin as strafication variable.

Rest, Sample = train_test_split(train_df, test_size = 0.8, stratify = train_df['fare_bin'])
#training the model with "Sample" data

model = sm.OLS(Sample.iloc[:,0], Sample.iloc[:,6:9].astype(float)).fit()
model.summary()
#making predictions of fare_amount using the variables in "Rest".

predictions_LR = model.predict(Rest.iloc[:,6:9])
#calculating different error metrics to check the accuracy of model

#Calculate MAPE

def MAPE(y_true, y_pred): 

    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100

    return mape

#Calculate MAPE

MAPE(Rest.iloc[:,0], predictions_LR)

#Calculate RMSE

def RMSE(y_true, y_pred): 

    rmse = np.sqrt(np.mean((y_true-y_pred)**2))

    return rmse

#Calculate RMSE

RMSE(Rest.iloc[:,0], predictions_LR)
#######################################   Decision Tree #############################################################

#traing the decision tree regression model

fit_DT = DecisionTreeRegressor(max_depth=5).fit(Sample.iloc[:,6:9], Sample.iloc[:,0])
fit_DT
#making predictions of fare_amount using the variables in "Rest".

predictions_DT = fit_DT.predict(Rest.iloc[:,6:9])
#Calculate MAPE

MAPE(Rest.iloc[:,0], predictions_DT)
#Calculate RMSE

RMSE(Rest.iloc[:,0], predictions_DT)
#######################################   Random Forest #############################################################

RF_model = RandomForestRegressor(n_estimators = 200).fit(Sample.iloc[:,6:9], Sample.iloc[:,0])
#making predictions of fare_amount using the variables in "Rest".

predictions_RF = RF_model.predict(Rest.iloc[:,6:9])
#Calculate MAPE

MAPE(Rest.iloc[:,0], predictions_RF)
#Calculate RMSE

RMSE(Rest.iloc[:,0], predictions_RF)
#######################################   KNN Regression #############################################################

KNN_model = KNeighborsRegressor(n_neighbors = 110).fit(Sample.iloc[:,6:9], Sample.iloc[:,0])
#making predictions of fare_amount using the variables in "Rest".

predictions_KNN = KNN_model.predict(Rest.iloc[:,6:9])
#Calculate MAPE

MAPE(Rest.iloc[:,0], predictions_KNN)
#making predictions of fare_amount on whole training data set

predictions_KNN2 = KNN_model.predict(train_df.iloc[:,6:9])
#Calculate MAPE of train_df

MAPE(train_df.iloc[:,0], predictions_KNN2)
#Calculate RMSE

RMSE(Rest.iloc[:,0], predictions_KNN)
#########################################################################################################################
#Since KNN has minimum MAPE value, it is accepted as the model to predict values of test.csv

# Load CSV

test = pd.read_csv("../input/test.csv")
#checking null values in the data

print(test.isnull().sum())
#adding diff_lat/long to test

add_travel_vector_features(test)
test.iloc[:,5:8]
test['fare_amount'] = KNN_model.predict(test.iloc[:,5:8])
test
# Writing a csv (output)

test.to_csv("test_new_py.csv", index = False)
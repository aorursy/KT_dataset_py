# Importing libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

plt.rcParams['figure.figsize'] = (8.0, 5.0)
df = pd.read_csv('../input/metro-bike-share-trip-data.csv')

df.head()
print("Shape of dataframe: ", df.shape)
# Checking for null values

df.isna().sum()
df = df.drop(columns=['Starting Lat-Long','Ending Lat-Long'])
df = df.dropna()
# Data Formatting

df['Start Time'] = pd.to_datetime(df['Start Time'])

df['End Time'] = pd.to_datetime(df['End Time'])

df['Passholder Type'] = df['Passholder Type'].astype('category')

df['Trip Route Category'] = df['Trip Route Category'].astype('category')
# Univariate graphs to see the distribution of Duration

df['Duration'].hist(figsize=(8,5))

plt.show()
# Correlation Matrix

sns.heatmap(df.drop(columns=['Trip ID','Starting Station ID','Ending Station ID','Bike ID']).corr(), annot=True)

plt.show()
round(df.describe(),2)
# Remove data where duration is less than 90 seconds and start station == send station

df = df.drop(df.index[(df['Duration'] < 90) & (df['Starting Station Latitude'] == df['Ending Station Latitude'])])
#Data for Top 5 Stations visual

top5 = pd.DataFrame()

top5['Station'] = df['Starting Station ID'].value_counts().head().index

top5['Number of Starts']=df['Starting Station ID'].value_counts().head().values

top5['Station'] = top5['Station'].astype('category')

top5['Station'] = top5.Station.cat.remove_unused_categories()
# Plot the top 5 stations

sns.barplot('Station', 'Number of Starts', data = top5)

plt.xticks(rotation=40, ha = 'right')

plt.title("Top 5 LA Metro Bike Stations by Number of Starts")

plt.show()
# Calculate trip duration based on tripduration(Seconds)

TD_user = pd.DataFrame()

TD_user['Avg. Trip Duration'] = round(df.groupby('Passholder Type')['Duration'].mean(),2)

TD_user = TD_user.reset_index()

TD_user['Passholder Type'] = TD_user['Passholder Type'].astype('object')
# Average Trip Duration by User Type (with anomalies)

g = sns.barplot('Passholder Type', 'Avg. Trip Duration', data = TD_user)

plt.Figure(figsize=(15,12))

plt.title("Average Trip Duration by User Type (with anomalies)")

for index, row in TD_user.iterrows():

    g.text(index,row['Avg. Trip Duration']-200,(str(row['Avg. Trip Duration'])+"  Seconds"), 

             color='white', ha="center", fontsize = 10)

plt.show()
#Boxplots are more informative to visualize breakdown of data

del(TD_user)



df.boxplot('Duration', by = 'Passholder Type')

plt.show()
#Add Minutes column for Trip Duration

df['Minutes'] = df['Duration']/60



#For Visual purposes, rounded

df['Minutes'] = round(df['Minutes'])

df['Minutes'] = df['Minutes'].astype(int)
# Calculate trip duration based on Minutes

TD_user2 = pd.DataFrame()

TD_user2['Avg. Trip Duration'] = round(df.groupby('Passholder Type')['Minutes'].mean(),1)

TD_user2 = TD_user2.reset_index()

TD_user2['Passholder Type'] = TD_user2['Passholder Type'].astype('object')
# Average Trip Duration by User Type based on Minutes

g = sns.barplot('Passholder Type', 'Avg. Trip Duration', data = TD_user2)

plt.Figure(figsize=(12,10))

plt.title("Average Trip Duration by User Type based on Minutes")

for index, row in TD_user2.iterrows():

    g.text(index,row['Avg. Trip Duration']-2,(str(row['Avg. Trip Duration'])+"  Minutes"), 

             color='white', ha="center", fontsize = 10)

plt.show()
del(TD_user2)



#Undo rounding for modelling purposes

df['Minutes'] = df['Duration']/60
trips_df = pd.DataFrame()

trips_df = df.groupby(['Starting Station ID','Ending Station ID']).size().reset_index(name = 'Number of Trips')

trips_df = trips_df.sort_values('Number of Trips', ascending = False)

trips_df['Starting Station ID'] = trips_df['Starting Station ID'].astype('str')

trips_df['Ending Station ID'] = trips_df['Ending Station ID'].astype('str')

trips_df["Trip"] = trips_df["Starting Station ID"] + " to " + trips_df["Ending Station ID"]

trips_df = trips_df[:10]

trips_df = trips_df.drop(['Starting Station ID', "Ending Station ID"], axis = 1)

trips_df = trips_df.reset_index()
# Most popular trips

g = sns.barplot('Number of Trips','Trip', data = trips_df)

plt.title("Most Popular Trips")

for index, row in trips_df.iterrows():

    g.text(row['Number of Trips']-50,index,row['Number of Trips'], 

             color='white', ha="center",fontsize = 10)

plt.show()
bike_use_df = pd.DataFrame()

bike_use_df = df.groupby(['Bike ID']).size().reset_index(name = 'Number of Times Used')

bike_use_df = bike_use_df.sort_values('Number of Times Used', ascending = False)

bike_use_df = bike_use_df[:10]

bike_use_df['Bike ID'] = bike_use_df['Bike ID'].astype(str)

bike_use_df['Bike ID'] = ('Bike ' + bike_use_df['Bike ID'])

bike_use_df = bike_use_df.reset_index()
#Visual of most used bike based on Number of Trips

g = sns.barplot('Number of Times Used','Bike ID', data = bike_use_df)

plt.title("Busiest Bike by Times Used")

for index, row in bike_use_df.iterrows():

    g.text(row['Number of Times Used']-30,index,row['Number of Times Used'], 

             color='white', ha="center",fontsize = 10)

plt.show()

#Bike usage based on minutes used

bike_min_df = pd.DataFrame()

bike_min_df['Minutes Used'] = df.groupby('Bike ID')['Minutes'].sum()

bike_min_df = bike_min_df.reset_index()

bike_min_df = bike_min_df.sort_values('Minutes Used', ascending = False)

bike_min_df['Bike ID'] = bike_min_df['Bike ID'].astype(str)

bike_min_df['Bike ID'] = ('Bike ' + bike_min_df['Bike ID'])

bike_min_df = bike_min_df[:10]

bike_min_df = bike_min_df.reset_index()
#Visual of most used bike based on minutesMost Popular Bikes by Minutes Used

g = sns.barplot('Minutes Used','Bike ID', data = bike_min_df)

plt.title("Busiest Bikes by Minutes Used")

for index, row in bike_min_df.iterrows():

    g.text(row['Minutes Used']-1500,index,round(row['Minutes Used'],2), 

             color='white', ha="center",fontsize = 10)

plt.show()
from datetime import datetime as dt

import calendar



df['Start Time'] = pd.to_datetime(df['Start Time'])

df['Start Time']= df['Start Time'].dt.date

df['start_day']=df['Start Time'].apply(lambda x:x.day)

df['start_month']=df['Start Time'].apply(lambda x:x.month)

df['start_day_of_week']=df['Start Time'].apply(lambda x:calendar.day_name[x.weekday()])



df['End Time'] = pd.to_datetime(df['End Time'])

df['End Time']= df['End Time'].dt.date

df['end_day']=df['End Time'].apply(lambda x:x.day)

df['end_month']=df['End Time'].apply(lambda x:x.month)

df['end_day_of_week']=df['End Time'].apply(lambda x:calendar.day_name[x.weekday()])
# Calculating the Distance between pickup and dropoff

from math import sin, cos, sqrt, atan2, radians



# calculating the distance using 'haversine' formula

def calculateDistance(row):

    R=6373.0 # approximate radius of earth in km

    pickup_lat=radians(row['Starting Station Latitude'])

    pickup_lon=radians(row['Starting Station Longitude'])

    dropoff_lat=radians(row['Ending Station Latitude'])

    dropoff_lon=radians(row['Ending Station Longitude'])

    dlon = dropoff_lon - pickup_lon

    dlat = dropoff_lat - pickup_lat

    a = sin(dlat / 2)**2 + cos(pickup_lat) * cos(dropoff_lat) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance
df['trip_distance'] = df.apply(lambda x:calculateDistance(x), axis=1)
def calculateBearing(lat1,lng1,lat2,lng2):

    R = 6371 

    lng_delta_rad = radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))

    y = sin(lng_delta_rad) * cos(lat2)

    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))
df['bearing']=df.apply(lambda row:calculateBearing(row['Starting Station Latitude'],row['Starting Station Longitude'],row['Ending Station Latitude'],row['Ending Station Longitude']),axis=1)
# Encoding days of week.

def encodeDays(day_of_week):

    day_dict={'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6}

    return day_dict[day_of_week]



df['start_day_of_week']=df['start_day_of_week'].apply(lambda x:encodeDays(x))

df['end_day_of_week']=df['end_day_of_week'].apply(lambda x:encodeDays(x))
# Encoding the categorical variables

df = pd.get_dummies(df, columns=['Passholder Type','start_day_of_week','end_day_of_week','start_month','end_month'])
drop_columns = ['Trip ID','Duration','Start Time','End Time','Starting Station ID','Starting Station Latitude','Starting Station Longitude','Ending Station ID','Ending Station Latitude','Ending Station Longitude','Bike ID','Trip Route Category','Plan Duration','Minutes']

tripids = df['Trip ID'].values

bikeids = df['Bike ID'].values

X = df.drop(drop_columns, axis=1)

y = df['Minutes'].values
# Scaling

from  sklearn.preprocessing  import StandardScaler



slc= StandardScaler()

X = slc.fit_transform(X)

y = slc.fit_transform(y.reshape(-1,1))
# Splitting

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
import lightgbm as lgb



lgb_params = {

    'learning_rate': 0.1,

    'max_depth': 8,

    'num_leaves': 55, 

    'objective': 'regression',

    'feature_fraction': 0.9,

    'bagging_fraction': 0.5,

    'max_bin': 300

}



y_train = y_train.ravel()

y_test = y_test.ravel()

dtrain = lgb.Dataset(X_train,y_train)



# Train a model

model_lgb = lgb.train(lgb_params, 

                      dtrain,

                      num_boost_round=1500)
from sklearn.metrics import mean_squared_error



pred_test = model_lgb.predict(X_test)

print("RMSE : ", mean_squared_error(pred_test,y_test))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split
import warnings
from sys import modules
import seaborn as sns
from datetime import datetime as dt
from datetime import timedelta
import math
import random
import time

# For transformations and predictions
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

# For the tree visualization
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO

# For scoring
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_squared_error as mse


# For validation
from sklearn.model_selection import train_test_split as split

import re

warnings.filterwarnings('ignore')
%matplotlib inline
n = 5819079 #number of records in file
s = 2000000 #desired sample size
rowsToSkip = sorted(random.sample(range(1,n),n-s))

flights_df = pd.read_csv('../input/flight-delays/flights.csv', \
                         parse_dates=['SCHEDULED_DEPARTURE'], \
                         skiprows=rowsToSkip)

delayed_only = flights_df[(flights_df['CANCELLED']==0) & (flights_df['DIVERTED']==0)]

delayed_only.drop(['DIVERTED','CANCELLED','CANCELLATION_REASON', \
                   'AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY', \
                   'LATE_AIRCRAFT_DELAY','WEATHER_DELAY','DEPARTURE_TIME','TAXI_OUT', \
                   'WHEELS_OFF', 'SCHEDULED_TIME','ELAPSED_TIME', \
                   'AIR_TIME','DISTANCE','WHEELS_ON','TAXI_IN', \
                   'SCHEDULED_ARRIVAL','ARRIVAL_TIME','ARRIVAL_DELAY'], axis=1, inplace=True)

delayed_only.columns = delayed_only.columns.str.lower()

print(delayed_only.shape)
delayed_only.head()
# Making the 'Scheduled_departure column to be a datetime type so we can use it in a real time variable

delayed_only['scheduled_departure_time'] = pd.to_datetime(delayed_only['scheduled_departure'],format="%H%M")
#delayed_only['scheduled_departure'] = delayed_only['scheduled_departure'].apply(lambda x: x.time())
delayed_only.head(10)
# Deciding on a level (=5 minutes) for a flight to be considered "delayed flight"

delayed_part = delayed_only[delayed_only['departure_delay']>0].shape[0]/delayed_only.shape[0]
print(f'The late departure part of all departures is: {delayed_part*100:.2f}%')

#delayed_only = delayed_only[delayed_only['departure_delay']>0]
delayed_only.shape
# delays < 5 = 0
#delayed_only['departure_delay'] = delayed_only['departure_delay'].where(delayed_only['departure_delay']>5.0, 0)
# with 0-5
delayed_only['departure_delay'] = delayed_only['departure_delay'].where(delayed_only['departure_delay']>0, 0)
# Combining 3 columns of day, month & year to a single date column
delayed_only['date']= delayed_only.apply(lambda x:dt.strptime("{0} {1} {2}".format(x['year'],x['month'], x['day']), "%Y %m %d"),axis=1)
# Adding a boolean column: late = True, on time/early = False
delayed_only['late_or_not'] = delayed_only['departure_delay'] > 0
lairport_df = pd.read_csv('../input/airports-codes/L_AIRPORT.csv')
lairport_df.rename(columns = {"Code":"iata_code"},inplace=True)
lairportid_df = pd.read_csv('../input/airports-codes/L_AIRPORT_ID.csv')
merged_airports_df = pd.merge(lairport_df,lairportid_df,on='Description')
merged_airports_df.set_index('Code',inplace=True)
merged_airports_df.sample(10)
airports_df = pd.read_csv('../input/flight-delays/airports.csv')
delayed_only['origin_airport'] = delayed_only['origin_airport'].astype(str)
airports_df.columns = airports_df.columns.str.lower()
print(airports_df.shape)
airports_df.head()
october = delayed_only[delayed_only['month']==10]
#october['origin_airport'] = delayed_only['origin_airport'].astype(str)
october['origin_airport'].replace({"11066":"CMH", "15016":"STL", "14730":"SDF", "12173":"HNL", "10157":"ACV", "15323":"TRI", "12758":"KOA", "15048":"SUX", "13158":"MAF", "10685":"BMI", "14543":"RKS", "15070":"SWF"},inplace=True) 
october['origin_airport'] = october.apply(lambda row: merged_airports_df.loc[int(row['origin_airport']),'iata_code'] if bool(re.match('[A-Z]+',row['origin_airport']))==False else row['origin_airport'], axis=1)
delayed_only = pd.concat([delayed_only[delayed_only['month']!=10],october])
delayed_only['origin_airport'] = delayed_only['origin_airport'].astype(str)
delayed_only = delayed_only[~delayed_only['origin_airport'].str.contains(" ")]
fig = plt.figure(figsize=(200,10))
ax = fig.gca()
delayed_only.groupby('origin_airport')['departure_delay'].mean().sort_values(ascending=False).plot.bar()
plt.show()
fig = plt.figure(figsize=(200,10))
ax = fig.gca()
delayed_only.groupby('origin_airport')['late_or_not'].mean().sort_values(ascending=False).plot.bar()
plt.show()

delayed_only['mean_airport_delay'] = delayed_only.groupby('origin_airport')['departure_delay'].transform('mean')
delayed_only['airport_delay_prcnt'] = delayed_only.groupby('origin_airport')['late_or_not'].transform('mean')
delayed_only['airport_delay_combined'] = delayed_only['mean_airport_delay'] * delayed_only['airport_delay_prcnt']
delayed_only.head()
fig = plt.figure(figsize=(23,23))
ax = airports_df.plot.scatter(x='longitude',y='latitude')
delayed_only.info()
# How many are late in general overview

plt.figure(figsize=(15,5))
ax = delayed_only['departure_delay'].plot.hist(bins=500)
ax.set_xlim(0,300)
plt.show()
log_delay = np.log1p(delayed_only.departure_delay)
#log_delay.hist(bins=1000)
delayed_only['log_delay'] = log_delay
delayed_only.head(20)
delayed_only['departure_delay'].describe(percentiles=[0.75,0.9,0.95,0.98,0.99])
#Checking the delays on each date.

# fig = plt.figure(figsize=(30,10))
# ax = fig.gca()
# delayed_only.plot.scatter(x='date', y='departure_delay',ax=ax)

# plt.show()
# check the size of the outliers we decide to drop.
max_delay=delayed_only['departure_delay'].quantile(0.98)
print(len(delayed_only[delayed_only['departure_delay']>max_delay])/(len(delayed_only))*100, max_delay)
#dropping outliers which are too unpredicted for our model
mask = (delayed_only['departure_delay'] > max_delay)
print(delayed_only.shape)
delayed_only = delayed_only.loc[~mask,:]
print(delayed_only.shape)
# dropping very low traffic airports as their score is not reliable

delayed_only['airport_flights'] = delayed_only.groupby('origin_airport')['origin_airport'].transform('count')
#my_scaler(delayed_only,'flights_number','scaled_flights')
min_airport=delayed_only.groupby('origin_airport').size().quantile(0.1)
print(delayed_only.shape)
delayed_only = delayed_only[delayed_only['airport_flights']>min_airport]
print(delayed_only.shape)
# filght dealys %    VS.   mean time of the delays

fig, ax1 = plt.subplots(figsize=(30,5))

ax1.plot(delayed_only.groupby('date')['late_or_not'].mean(),data=delayed_only, color='g')
ax1.set_xlabel('Days of the year')
ax1.set_ylabel('late %', color='g')
ax1.set_xlim(min(delayed_only['date']), max(delayed_only['date']))

ax2 = ax1.twinx()
ax2.plot(delayed_only.groupby('date')['departure_delay'].mean(),data=delayed_only, color = "r")
#ax2.plot(delayed_only.groupby('date')['log_delay'].mean(),data=delayed_only, color = "r")
ax2.set_ylabel('Time of delay', color = "r")

plt.show()
#delayed_only.groupby('date')['log_delay'].transform('mean')
delayed_only.groupby('date')['departure_delay'].transform('mean')
#delayed_only['mean_date_delay'] = delayed_only.groupby('date')['log_delay'].transform('mean')
delayed_only['mean_date_delay'] = delayed_only.groupby('date')['departure_delay'].transform('mean')
delayed_only['mean_date_delay'].hist(bins=50)

delayed_only['mean_date_delay'].describe(percentiles=[0.75,0.9,0.95,0.98,0.99])
sns.lineplot(data=delayed_only, x='date', y='mean_date_delay')
plt.figure(figsize=(40,15))
delayed_only.groupby('date')['departure_delay'].mean().plot()
#chosing the mean date delay limit to be considered as busy day 
delayed_only['busy_day']=delayed_only['mean_date_delay']>13
# Late rate and mean delay by Airline

fig = plt.figure(figsize=(20,6))
ax1 = fig.gca()

#delayed_only.groupby('airline')['log_delay','late_or_not'].mean().sort_values(by='log_delay', ascending=False).plot(kind='bar', secondary_y='late_or_not', ax=ax1)
delayed_only.groupby('airline')['departure_delay','late_or_not'].mean().sort_values(by='departure_delay', ascending=False).plot(kind='bar', secondary_y='late_or_not', ax=ax1)
plt.show()
delayed_only.groupby('airline')['late_or_not'].mean().sort_values()
# Boolean or dummies ?
#delayed_only.groupby('airline')['log_delay'].mean().sort_values()
delayed_only.groupby('airline')['departure_delay'].mean().sort_values()
#delayed_only['good_airline']=delayed_only[delayed_only['departure_delay'].mean()<13]
#delayed_only['mean_airline_log_delay'] = delayed_only.groupby('airline')['log_delay'].transform('mean')
delayed_only['mean_airline_delay'] = delayed_only.groupby('airline')['departure_delay'].transform('mean')
delayed_only['airline_delay_prcnt'] = delayed_only.groupby('airline')['late_or_not'].transform('mean')
delayed_only['airline_delay_combined'] = delayed_only['mean_airline_delay'] * delayed_only['airline_delay_prcnt']
#delayed_only['good_airline']=delayed_only['mean_airline_delay']<0.85
delayed_only['good_airline']=delayed_only['mean_airline_delay']<10

delayed_only.head()
# Late rate and mean late by day of week

fig = plt.figure(figsize=(10,6))
ax1 = fig.gca()

#delayed_only.groupby('day_of_week')['log_delay','late_or_not'].mean().plot(kind='bar', secondary_y='late_or_not', ax=ax1)
delayed_only.groupby('day_of_week')['departure_delay','late_or_not'].mean().plot(kind='bar', secondary_y='late_or_not', ax=ax1)
plt.show()
# airline with day of the week check. in most of the airlines we can see a rise on weekends.
fig, axes = plt.subplots(2, tight_layout=True, figsize=(20,10))

delayed_only.groupby(['airline','day_of_week'])['late_or_not'].mean().unstack().plot.bar(ax=axes[0], title='Late % across airlines')
#delayed_only.groupby(['airline','day_of_week'])['log_delay'].mean().unstack().plot.bar(ax=axes[1], title='Late in time across airlines')
delayed_only.groupby(['airline','day_of_week'])['departure_delay'].mean().unstack().plot.bar(ax=axes[1], title='Late in time across airlines')
plt.show()
#delayed_only.groupby(pd.Grouper(key='scheduled_departure',freq='H'))['log_delay'].mean().sort_values()
delayed_only.groupby(pd.Grouper(key='scheduled_departure_time',freq='H'))['departure_delay'].mean().sort_values()
#delayed_only.groupby(pd.Grouper(key='scheduled_departure_time',freq='H'))['log_delay'].mean().plot.bar()
delayed_only.groupby(pd.Grouper(key='scheduled_departure_time',freq='H'))['departure_delay'].mean().plot.bar()
#delayed_only['hour_avg_delay'] = delayed_only.groupby(pd.Grouper(key='scheduled_departure',freq='H'))['log_delay'].transform('mean')
delayed_only['hour_avg_delay'] = delayed_only.groupby(pd.Grouper(key='scheduled_departure_time',freq='H'))['departure_delay'].transform('mean')
#delayed_only['rush_hours']=delayed_only.hour_avg_delay>10
delayed_only.head(10)
delayed = delayed_only[delayed_only['departure_delay']!=0]
delayed.shape
fig = plt.figure(figsize=(30,5))
ax = fig.gca()
delayed.groupby('scheduled_departure')['departure_delay'].median().plot(title='Delays length  across hours of the day', ax=ax)
ax.set_ylim(0,120)
plt.show()
# taking only the hour int of the scheduled_departure for the model
delayed_only['scheduled_hour_int']=delayed_only['scheduled_departure'].apply(lambda x: int(x[0:2]))
delayed_only.head()
def my_scaler(df, from_col_name, to_col_name):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(pd.DataFrame(df[from_col_name],index=df.index))
    scaled_df = pd.DataFrame(scaled, columns=['scale_temp'], index=df.index)
    df[to_col_name] = scaled_df['scale_temp']
    
    
# scaler = MinMaxScaler()
# scaled_delay = scaler.fit_transform(pd.DataFrame(delayed_only['mean_airline_delay'],index=delayed_only.index))
# scaled_df = pd.DataFrame(scaled_delay, columns=['mean_airline'], index=delayed_only.index)
# delayed_only['scaled_airline'] = scaled_df['mean_airline']
# Scaling 'airline_delay_combined' & 'airport_delay_combined'
my_scaler(delayed_only,'airline_delay_combined','scaled_airline_score')
my_scaler(delayed_only,'airport_delay_combined','scaled_airport_score')
holidays_df = pd.read_csv('../input/us-holidays/usholidays.csv', parse_dates=['Date'])
print(holidays_df.shape)
holidays_df.info()
holidays_df = holidays_df[holidays_df['Date'].dt.year==2015]
holidays_list = holidays_df['Date'].dt.date.tolist()
holidays = []
for holiday in holidays_list:
    holidays.append(holiday)
    for i in range(1,4):
        holidays.append(holiday+timedelta(days=i))
        holidays.append(holiday-timedelta(days=i))

len(holidays)
delayed_only['is_holiday'] = delayed_only['date'].isin(holidays)
delayed_only.head()
len(delayed_only[delayed_only['is_holiday']==True])
delayed_only.groupby('is_holiday')['departure_delay'].mean().plot.bar()
relevant_cols = ['mean_date_delay','scaled_airline_score', 'scheduled_hour_int','airport_flights', 'scaled_airport_score', 'is_holiday']
X = delayed_only.loc[:,relevant_cols]
#y = delayed_only['log_delay']
y = delayed_only['departure_delay']
X.head()
print(X.shape)
y.head()
X_train, X_test, y_train, y_test = split(X, y, random_state=314159)
fd_model = DecisionTreeRegressor(max_leaf_nodes=120, min_samples_leaf=1000).fit(X_train, y_train)
def visualize_tree(model, md=5):
    dot_data = StringIO()  
    export_graphviz(model, out_file=dot_data, feature_names=X_train.columns, max_depth=md)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]  
    return Image(graph.create_png(), width=800) 
visualize_tree(fd_model)
y_train_pred = fd_model.predict(X_train)

ax = sns.scatterplot(x=y_train, y=y_train_pred)
ax.set_ylim(0,max_delay)
ax.set_xlim(0,max_delay)
ax.set_ylabel('prediction')
ax.plot(y_train, y_train, 'r')
for feature, importance in zip(X.columns, fd_model.feature_importances_):
    print(f'{feature:14}: {importance:.2f}')
RMSLE = msle(y_train, y_train_pred)**0.5
RMSLE
prediction = pd.DataFrame({'y': y_train, 'y_pred': y_train_pred})
prediction.sample(10)
y_test_pred = fd_model.predict(X_test)
ax = sns.scatterplot(x=y_test, y=y_test_pred)
ax.set_ylim(0,max_delay)
ax.set_xlim(0,max_delay)
ax.plot(y_test, y_test, 'r')
RMSLE = msle(y_test, y_test_pred)**0.5
RMSLE
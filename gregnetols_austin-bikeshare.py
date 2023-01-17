# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk # SkLearn ML library

from sklearn.model_selection import train_test_split

from sklearn import linear_model

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import requests

from datetime import datetime
trips = pd.read_csv('../input/austin-bike/austin_bikeshare_trips.csv')

stations = pd.read_csv('../input/austin-bike/austin_bikeshare_stations.csv')
trips.head(5)
trips.dtypes
stations.head()
tripsByMonth = trips.groupby('month').month.count()

tripsByMonth.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',

                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']



ax = sns.barplot(x='index', y='month', data=tripsByMonth.reset_index(), color='red')

ax.figure.set_size_inches(14,8)

sns.set_style(style='white')

ax.axes.set_title('Total Rides in Each Month', fontsize=24)

ax.set_xlabel('Month', size=20)

ax.set_ylabel('Rides', size=20)

ax.tick_params(labelsize=16)

tripsByYearMonth = trips

tripsByYearMonth = tripsByYearMonth.groupby(['month','year']).month.count()

tripsByYearMonth
tripsFullYears = trips[trips['year'].isin(['2014','2015'])]

tripsByMonth = tripsFullYears.groupby(['month', 'year']).trip_id.count()





ax = sns.barplot(x='month', y='trip_id', hue='year', data=tripsByMonth.reset_index(), color='red')

ax.figure.set_size_inches(14,8)

sns.set_style(style='white')

ax.axes.set_title('Total Rides in Each Month', fontsize=24)

ax.set_xlabel('Month', size=20)

ax.set_ylabel('Rides', size=20)

ax.tick_params(labelsize=16)
#Create a binary column for trips that start and end at the same station

def round_trip(row):

    if row['end_station_id'] == row['start_station_id']:

        return 1

    return 0



trips['round_trip'] = trips.apply(lambda row: round_trip(row), axis=1)
aggregate = {'trip_id':'count', 'round_trip':'sum'}

roundTripsByMonth = trips.groupby('month').agg(aggregate)

roundTripsByMonth['round_trip_ratio'] = roundTripsByMonth['round_trip'] / roundTripsByMonth['trip_id'] * 100



#Replace float monthes with string months

roundTripsByMonth.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',

                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']



ax = sns.barplot(x='index', y='round_trip_ratio', data=roundTripsByMonth.reset_index(), color='red')

ax.figure.set_size_inches(14,8)

ax.set_ylim(0,20)

ax.axes.set_title('Percent of Rides That Are Return Trips', fontsize=24)

ax.set_xlabel('Month', size=20)

ax.set_ylabel('Percent', size=20)

ax.tick_params(labelsize=16)



def short_term_subscriber(row):

    if (

            row['subscriber_type'].lower().find('walk') > -1 or

            row['subscriber_type'].lower().find('weekend') > -1 or

            row['subscriber_type'].lower().find('24') > -1 or

            row['subscriber_type'].lower().find('single') > -1

        ):

        return 1

    return 0



trips['subscriber_type'] = trips['subscriber_type'].replace(np.nan, '', regex=True)

trips['short_term_membership'] = trips.apply(lambda row: short_term_subscriber(row), axis=1)
trips.head()
aggregate = {'trip_id': 'count', 'short_term_membership': 'sum'}

membershipTypeTripsPerMonth = trips.groupby('month').agg(aggregate)

membershipTypeTripsPerMonth['short_term_membership_percentage'] = membershipTypeTripsPerMonth['short_term_membership'] / membershipTypeTripsPerMonth['trip_id'] * 100 



membershipTypeTripsPerMonth.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',

                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']



ax = sns.barplot(x='index', y='short_term_membership_percentage', data=membershipTypeTripsPerMonth.reset_index(), color='red')

ax.figure.set_size_inches(14,8)

ax.axes.set_title('Percent of Rides Using a Short Term Membership', fontsize=24)

ax.set_xlabel('Months', fontsize=20)

ax.set_ylabel('Percentage', fontsize=20)

ax.tick_params(labelsize=16)
weather = pd.read_csv('../input/austin-weather/austin_weather.csv')

trips = pd.read_csv('../input/austin-bike/austin_bikeshare_trips.csv')

stations = pd.read_csv('../input/austin-bike/austin_bikeshare_stations.csv')
weather.head()
weather.Events.unique()
weather['Rain'] = np.where(weather['Events'].str.contains('Rain'), 1, 0)

weather['Thunderstorm'] = np.where(weather['Events'].str.contains('Thunderstorm'), 1, 0)

weather['Fog'] = np.where(weather['Events'].str.contains('Fog'), 1, 0)

weather['Snow'] = np.where(weather['Events'].str.contains('Snow'), 1, 0)



weather = weather.drop('Events', 1)
weather['PrecipitationSumInches'] = np.where(weather['PrecipitationSumInches'] == 'T', 0.001, weather['PrecipitationSumInches'])
weather['Date'] = pd.to_datetime(weather['Date'])

weather['DayOfWeek'] = weather['Date'].dt.weekday
weather = weather[(weather['Date'] >= '2014-01-01') & (weather['Date'] <= '2015-12-31')]
weather = weather.set_index('Date', drop=True)
weather.head()
weather = weather.convert_objects(convert_numeric=True)

weather = weather.fillna(weather.mean())
weather.dtypes
trips.head()
trips = trips[trips['year'].isin(['2014','2015'])]

trips['Date'] = pd.to_datetime(trips['start_time']).dt.date

trips = trips.groupby(['Date']).trip_id.count()
trips.name = 'TripCount'
trips.shape
tripWeather = trips.to_frame().join(weather, lsuffix='Date', rsuffix='Date', how='inner')
rideCounts = tripWeather['TripCount']

rideWeather = tripWeather.drop('TripCount', axis=1)
rideWeather_train, rideWeather_test, rideCounts_train, rideCounts_test = train_test_split( 

    rideWeather, rideCounts, test_size = .3, random_state = 13, shuffle=True)
rideWeather_train.shape
rideCounts_train.shape
rideWeather_test.shape
rideCounts_test.shape
reg = linear_model.Ridge (alpha = .5)
reg.fit(rideWeather_train, rideCounts_train)
ridgeScore = reg.score(rideWeather_test, rideCounts_test)

print(ridgeScore)
rideCountsPredictions = reg.predict(rideWeather_test)

rideCountsActual = rideCounts_test.as_matrix()
ax = sns.regplot(x=rideCountsActual, y=rideCountsPredictions)

ax.figure.set_size_inches(10,6)

ax.axes.set_title('Predictions Vs. Actual', fontsize=24)

ax.set_xlabel('Actual', fontsize=20)

ax.set_ylabel('Predictions', fontsize=20)

ax.tick_params(labelsize=16)
weather.head(10)
#Calculate correleation matrix

correlation = tripWeather.corr()



# plot the heatmap

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap( correlation, ax=ax)
def is_weekend(row):

    if row >= 5:

        return 1

    return 0



weather['Weekend'] = weather['DayOfWeek'].apply(is_weekend)

weather = weather.drop('DayOfWeek', axis=1)
weather.head()
weather = weather.drop('TempLowF', axis=1)
weather = weather.drop(['DewPointHighF', 'DewPointLowF'], axis=1)
weather = weather.drop(['HumidityHighPercent', 'HumidityLowPercent'], axis=1)
weather['PressureChange'] = weather['SeaLevelPressureHighInches'] - weather['SeaLevelPressureLowInches']



weather = weather.drop(['SeaLevelPressureHighInches', 'SeaLevelPressureAvgInches', 'SeaLevelPressureLowInches'], axis=1)
weather = weather.drop(['VisibilityHighMiles', 'VisibilityLowMiles'], axis=1)
weather = weather.drop(['WindHighMPH', 'WindGustMPH'], axis=1)
weather.head()
tripWeather = trips.to_frame().join(weather, lsuffix='Date', rsuffix='Date', how='inner')
rideCounts = tripWeather['TripCount']

rideWeather = tripWeather.drop('TripCount', axis=1)



rideWeather_train, rideWeather_test, rideCounts_train, rideCounts_test = train_test_split( 

    rideWeather, rideCounts, test_size = .3, random_state = 13, shuffle=True)
print(rideWeather_train.shape)

print(rideWeather_test.shape)

print(rideCounts_train.shape)

print(rideCounts_test.shape)
reg = linear_model.Ridge (alpha = .5)
reg.fit(rideWeather_train, rideCounts_train)
ridgeScore = reg.score(rideWeather_test, rideCounts_test)

print(ridgeScore)
rideCountsPredictions = reg.predict(rideWeather_test)

rideCountsActual = rideCounts_test.as_matrix()
ax = sns.regplot(x=rideCountsActual, y=rideCountsPredictions)

ax.figure.set_size_inches(10,6)

ax.axes.set_title('Predictions Vs. Actual', fontsize=24)

ax.set_xlabel('Actual', fontsize=20)

ax.set_ylabel('Predictions', fontsize=20)

ax.tick_params(labelsize=16)
tripWeather = trips.to_frame().join(weather, lsuffix='Date', rsuffix='Date', how='inner')



rideCounts = tripWeather['TripCount']

rideWeather = tripWeather.drop('TripCount', axis=1)



rideWeather_train, rideWeather_test, rideCounts_train, rideCounts_test = train_test_split( 

    rideWeather, rideCounts, test_size = .3, random_state = 13, shuffle=True)



print(rideWeather_train.shape)

print(rideWeather_test.shape)

print(rideCounts_train.shape)

print(rideCounts_test.shape)
#initialize the model

reg = linear_model.Lasso(alpha=0.1)



#fit the training data to the model

reg.fit(rideWeather_train, rideCounts_train)



#find the R^2 score for the results

LassoScore = reg.score(rideWeather_test, rideCounts_test)

print(LassoScore)
rideCountsPredictions = reg.predict(rideWeather_test)

rideCountsActual = rideCounts_test.as_matrix()



ax = sns.regplot(x=rideCountsActual, y=rideCountsPredictions)

ax.figure.set_size_inches(10,6)

ax.axes.set_title('Predictions Vs. Actual', fontsize=24)

ax.set_xlabel('Actual', fontsize=20)

ax.set_ylabel('Predictions', fontsize=20)

ax.tick_params(labelsize=16)
outlierTrips = trips[trips > 1500]

outlierTrips
weather.head()
def spring_break_woo(date):

    if (date >= datetime(2015, 3, 14)) & (date <= datetime(2015, 3, 23)):

        return 1

    if (date >= datetime(2014, 3, 8)) & (date <= datetime(2014, 3, 17)):

        return 1

    return 0



weather['Date'] = weather.index

weather['SpringBreak'] = weather['Date'].apply(spring_break_woo)

weather = weather.drop('Date', axis=1)
tripWeather = trips.to_frame().join(weather, lsuffix='Date', rsuffix='Date', how='inner')



rideCounts = tripWeather['TripCount']

rideWeather = tripWeather.drop('TripCount', axis=1)



rideWeather_train, rideWeather_test, rideCounts_train, rideCounts_test = train_test_split( 

    rideWeather, rideCounts, test_size = .3, random_state = 13, shuffle=True)



print(rideWeather_train.shape)

print(rideWeather_test.shape)

print(rideCounts_train.shape)

print(rideCounts_test.shape)
#initialize the model

reg = linear_model.Lasso(alpha=0.1)



#fit the training data to the model

reg.fit(rideWeather_train, rideCounts_train)



#find the R^2 score for the results

LassoScore = reg.score(rideWeather_test, rideCounts_test)

print(LassoScore)
rideCountsPredictions = reg.predict(rideWeather_test)

rideCountsActual = rideCounts_test.as_matrix()



ax = sns.regplot(x=rideCountsActual, y=rideCountsPredictions)

ax.figure.set_size_inches(10,6)

ax.axes.set_title('Predictions Vs. Actual', fontsize=24)

ax.set_xlabel('Actual', fontsize=20)

ax.set_ylabel('Predictions', fontsize=20)

ax.tick_params(labelsize=16)
reg = linear_model.Ridge(alpha = .5)



reg.fit(rideWeather_train, rideCounts_train)



ridgeScore = reg.score(rideWeather_test, rideCounts_test)

print(ridgeScore)
rideCountsPredictions = reg.predict(rideWeather_test)

rideCountsActual = rideCounts_test.as_matrix()



ax = sns.regplot(x=rideCountsActual, y=rideCountsPredictions)

ax.figure.set_size_inches(10,6)

ax.axes.set_title('Predictions Vs. Actual', fontsize=24)

ax.set_xlabel('Actual', fontsize=20)

ax.set_ylabel('Predictions', fontsize=20)

ax.tick_params(labelsize=16)
 # This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

color = sns.color_palette()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



orig = pd.read_csv("../input/nyc-taxi-trip-duration/train.csv")

extra1 = pd.read_csv("../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv")

extra2 = pd.read_csv("../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv")

extra = pd.concat([extra1, extra2])

train = pd.merge(orig, extra, how='left', on='id')

orig = pd.read_csv("../input/nyc-taxi-trip-duration/test.csv")

extra = pd.read_csv("../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv")

test = pd.merge(orig, extra, how='left', on='id')
train.head(3)
test.head(3)
train['log_trip_duration'] = np.log(train['trip_duration']+1)

plt.figure(figsize=(10, 5)) 

ax = plt.axes()

sns.distplot(train['log_trip_duration'], ax=ax)

ax.set_xlabel('log_trip_duration')

ax.set_ylabel('number of records')

ax.set_title('Distribution')
city_long_border = (-74.03, -73.75)

city_lat_border = (40.63, 40.85)

sns.jointplot(x='pickup_longitude', y='pickup_latitude', data=train.loc[:100000,:], kind='scatter', 

              xlim=city_long_border, ylim=city_lat_border, s=1).set_axis_labels('longitude', 'latitude')

sns.plt.show()

sns.jointplot(x='dropoff_longitude', y='dropoff_latitude', data=train.loc[:100000,:], kind='scatter', 

              xlim=city_long_border, ylim=city_lat_border, s=1).set_axis_labels('longitude', 'latitude')

sns.plt.show()
sns.jointplot(x='pickup_longitude', y='pickup_latitude', data=test.loc[:100000,:], kind='scatter', 

              xlim=city_long_border, ylim=city_lat_border, s=1).set_axis_labels('longitude', 'latitude')

sns.plt.show()

sns.jointplot(x='dropoff_longitude', y='dropoff_latitude', data=test.loc[:100000,:], kind='scatter', 

              xlim=city_long_border, ylim=city_lat_border, s=1).set_axis_labels('longitude', 'latitude')

sns.plt.show()
# may traffic light has some influences on trip duration

def extractNumberOfTurn(data):

    # extract the number of turns in recommand path

    data['left'] = data['step_direction'].apply(lambda x: sum([i == 'left' for i in str(x).split('|')]))

    data['right'] = data['step_direction'].apply(lambda x: sum([i == 'right' for i in str(x).split('|')]))

    return data



train = extractNumberOfTurn(train)

test = extractNumberOfTurn(test)
def dateExtract(data):

    data['pickup_datetime'] = pd.to_datetime(data.pickup_datetime)

    data['pickup_date'] = data['pickup_datetime'].dt.date

    data['pickup_hour'] = data['pickup_datetime'].dt.hour

    data['pickup_weekday'] = data['pickup_datetime'].dt.weekday + 1

    data['is_workday'] = data['pickup_weekday'] < 6

    

    data['store_and_fwd_flag'] = data['store_and_fwd_flag'].map({'N':0, 'Y': 1}).astype(int)

    return data



train = dateExtract(train)

test = dateExtract(test)
from matplotlib.ticker import MultipleLocator 

from matplotlib.dates import DateFormatter

with sns.axes_style():

    fig = plt.figure(figsize=(18, 8))

    ax = sns.plt.axes()

    draw_data = train.groupby(['pickup_date'])[['id']].count()

    plt.plot(draw_data, 'o-', label='train')

    draw_data = test.groupby(['pickup_date'])[['id']].count()

    plt.plot(draw_data, 'o-', label='test')

    plt.legend(loc=0)

    ax.xaxis.set_major_locator(MultipleLocator(7))   # set the value of axis x to weekly

    ax.xaxis.set_major_formatter(DateFormatter( '%Y-%m-%d' ))   # control the date format

    fig.autofmt_xdate()

    plt.xticks(fontsize=15)

    plt.yticks(fontsize=15)

    sns.plt.show()
def Haversine(lat1, long1, lat2, long2):

    lat1, long1, lat2, long2 = map(np.radians, (lat1, long1, lat2, long2))

    dLat = lat2 - lat1

    dLong = long2 -long1

    a = np.sin(dLat*0.5)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLong*0.5)**2  

    c = 2 * np.arcsin(np.sqrt(a))  # sphere centre angle between point 1 and 2

    r = 6371 # the average radius of earth 

    return c * r





def Manhattan(lat1, long1, lat2, long2):

    a = Haversine(lat1, long1, lat1, long2)

    b = Haversine(lat1, long2, lat2, long2)

    return a + b





def distanceInfo(data):

    # get the space distance

    data['Haversine'] = Haversine(data['dropoff_latitude'], data['dropoff_longitude'], 

                                  data['pickup_latitude'], data['pickup_longitude'])

    # get the L1 distance (cab distance)

    data['Manhattan'] = Manhattan(data['dropoff_latitude'], data['dropoff_longitude'], 

                                  data['pickup_latitude'], data['pickup_longitude'])

    

    data['span_latitude'] = data['dropoff_latitude'] - data['pickup_latitude']

    data['span_longitude'] = data['dropoff_longitude'] - data['pickup_longitude']

    data['angle'] = np.fabs(data['span_latitude'] / data['span_longitude'])

    data.loc[data['angle']<1, 'angle'] = 1 / data.loc[data['angle']<1, 'angle']

    return data





train = distanceInfo(train)

test = distanceInfo(test)

train['average_speed'] = train['Manhattan'] / train['trip_duration'] * 3600

print(train.loc[:5, ['Haversine', 'Manhattan', 'average_speed', 'angle', 'trip_duration']])
fig, ax = plt.subplots(ncols=2, sharey=False, figsize=(12, 5))

combine = pd.concat([train.drop(['trip_duration', 'dropoff_datetime'], 1), test])

draw_data = combine.groupby(['pickup_weekday'])['id'].count()

ax[0].plot(draw_data, 'bo-')

ax[0].set_xlabel('weekday')

ax[0].set_ylabel('number of orders')

draw_data = combine.groupby(['pickup_hour'])['id'].count()

ax[1].plot(draw_data, 'bo-')

ax[1].set_xlabel('hour')

plt.show()

fig, ax = plt.subplots(ncols=2, sharey=False, figsize=(12, 5))

draw_data = train.groupby(['pickup_weekday'])['average_speed'].mean()

ax[0].plot(draw_data, 'bo-')

ax[0].set_xlabel('weekday')

ax[0].set_ylabel('average_speed')

draw_data = train.groupby(['pickup_hour'])['average_speed'].mean()

ax[1].plot(draw_data, 'bo-')

ax[1].set_xlabel('hour')

plt.show()
with sns.axes_style():

    fig = plt.figure(figsize=(18, 8))

    ax = sns.plt.axes()

    draw_data = train.groupby('pickup_date')[['average_speed']].mean()

    plt.plot(draw_data, 'o-', label='speed')

    draw_data = train.groupby('pickup_date')[['id']].count()

    plt.plot(draw_data/200, 'o-', label='order')

    plt.legend(loc=0)

    ax.xaxis.set_major_locator(MultipleLocator(7))   # set the value of axis x to weekly

    ax.xaxis.set_major_formatter(DateFormatter( '%Y-%m-%d' ))   # control the date format

    fig.autofmt_xdate()

    plt.xticks(fontsize=15)

    plt.yticks(fontsize=15)

    sns.plt.show()
data = train.groupby(['pickup_date', 'pickup_hour'])[['average_speed']].mean()

print(data[:10])
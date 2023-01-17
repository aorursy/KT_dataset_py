# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pytz as tz

import matplotlib.pyplot as plt



from datetime import datetime

from sklearn.cluster import KMeans



from sklearn.metrics import silhouette_score
df = pd.read_csv('../input/gowalla-checkins/Gowalla_totalCheckins.txt', sep='\t', header=None)

df.columns = ['userid','timestamp','latitude','longitude','spotid']

df.head()
lon_min, lat_min, lon_max, lat_max = -74.2589, 40.4774, -73.7004, 40.9176

nyc_events = df[(df['longitude']>lon_min) & 

           (df['longitude']<lon_max) & 

           (df['latitude']>lat_min) & 

           (df['latitude']<lat_max)]

nyc_events.head()
nyc_events.shape
venues = pd.read_csv('../input/venues-in-new-york-city/spots.txt', sep='\t', header=0)

venues.head()
def parse_datetime(s):

    tzone = tz.timezone("America/New_York") #parse_datetime

    utc = datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ')

    return tz.utc.localize(utc).astimezone(tzone)
nyc_events['ts'] = nyc_events['timestamp'].apply(lambda x: parse_datetime(x))

nyc_events = nyc_events.drop('timestamp',axis=1,errors='ignore')



#local date and time

nyc_events['date']  = nyc_events['ts'].astype(object).apply(lambda x : x.date())

nyc_events['time']  = nyc_events['ts'].astype(object).apply(lambda x : x.time())



#day of the week (localtime)

#hour of the day (localtime)

nyc_events['weekday']  = nyc_events['date'].astype(object).apply(lambda x : x.weekday())

nyc_events['day']   = nyc_events['date'].astype(object).apply(lambda x : x.day)

nyc_events['hour']   = nyc_events['time'].astype(object).apply(lambda x : x.hour)



nyc_events[['ts','date','time','weekday','day','hour']][0:5]
nyc_events = pd.DataFrame.merge(nyc_events, venues[['spotid','spotname']], on='spotid', how="inner")

nyc_events.head()
# Let us plot the events registered, ordered temporally by increasing timestamps and grouped by date

plt.rcParams['figure.figsize'] = (20.0, 20.0)

plt.rcParams.update({'font.size': 12})

plt.rcParams['xtick.major.pad']='5'

plt.rcParams['ytick.major.pad']='5'

plt.style.use('ggplot')



plt.subplot(4, 1, 1)

top = nyc_events.groupby('date').size()

plt.plot(top.index, top, 'g-')

plt.title('Gowalla App usage over time')

plt.show()
plt.style.use('ggplot')

fig = plt.figure()

fig.set_size_inches(21,4)



plt.subplot(1, 3, 1)

top = nyc_events.groupby('hour').size()

plt.bar(top.index, top,align='center', color='red')

plt.title('Check-ins: Hour of the day')



plt.subplot(1, 3, 2)

top = nyc_events.groupby('weekday').size()

plt.bar(top.index, top,align='center',color='blue')

plt.title('Check-ins: Day of the week')



plt.subplot(1, 3, 3)

top = nyc_events.groupby('day').size()

plt.bar(top.index, top,align='center',color='green')

plt.title('Check-ins: Day of the month')



plt.show()
#for k in range(200, 270, 10):

    #kmeans = KMeans(n_clusters=k, init='k-means++')

    #X_sample = (nyc_events[['longitude','latitude']].sample(frac=0.1))

    #kmeans.fit(X_sample)

    #y = kmeans.labels_

    #print("k =", k, " silhouette_score ", silhouette_score(X_sample, y, metric='euclidean'))



# Rule of thumb for k: sqrt(n/2); here n is 112390 - total no. of NYC events

kmeans = KMeans(n_clusters=240, init='k-means++')



# Compute the clusters based on longitude and latitude features

X_sample = nyc_events[['longitude','latitude']].sample(frac=0.1)

kmeans.fit(X_sample)

y = kmeans.labels_

print("k = 240", " silhouette_score ", silhouette_score(X_sample, y, metric='euclidean'))
nyc_events['cluster'] = kmeans.predict(nyc_events[['longitude','latitude']])

nyc_events[['userid','latitude','longitude','spotid','cluster']].sample(10)
gdf = nyc_events.groupby(['cluster', 'spotid']).size().reset_index()

gdf.columns = ['cluster', 'spotid', 'count']

idx = gdf.groupby(['cluster'])['count'].transform(max) == gdf['count']

topvenues_df = gdf[idx].merge(venues, on='spotid', how='left').sort_values(by='count', ascending=False)

#top 10 out of 200 clusters by events count

topvenues = topvenues_df[:10]
# Most Visited Venues

plt.style.use('ggplot')

fig = plt.figure()

fig.set_size_inches(21,5)



plt.bar(range(len(topvenues)), topvenues['count'], align='center')

plt.xticks(range(len(topvenues)),topvenues['spotname'], rotation='vertical')

plt.title('Most visited venues')

plt.show()
def recommend_venues(df, longitude, latitude):

    predicted_cluster = kmeans.predict(np.array([longitude,latitude]).reshape(1,-1))[0]

    # Fetch the venue name of the top most record in the topvenues dataframe for the predicted cluster

    venue_name = df[df['cluster']==predicted_cluster].iloc[0]['spotname']

    msg = 'What about visiting the ' + venue_name + '?'

    return msg
recommend_venues(topvenues_df, -74, 40.55)
recommend_venues(topvenues_df, -73.993, 40.75)
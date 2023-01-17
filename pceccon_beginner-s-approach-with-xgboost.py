import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import geojson

import xgboost as xgb

import warnings

warnings.filterwarnings('ignore')



from mpl_toolkits.basemap import Basemap

from matplotlib import cm

from descartes import PolygonPatch

from haversine import haversine



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import MiniBatchKMeans



%matplotlib inline
from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))
df = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')

df['pickup_datetime']  = pd.to_datetime(df['pickup_datetime'])

df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])



df.head()
hf = pd.read_csv('../input/nyc2016holidays/NYC_2016Holidays.csv', sep=';')



hf['Date'] = hf['Date'].apply(lambda x: x + ' 2016')

hf['Date']  = pd.to_datetime(hf['Date'])

hf
wf = pd.read_csv('../input/weather-data-in-new-york-city-2016/weather_data_nyc_centralpark_2016.csv')

wf['date']  = pd.to_datetime(wf['date'])



wf.head()
min_lat = df['pickup_latitude'].min()

max_lat = df['pickup_latitude'].max()

min_lon = df['pickup_longitude'].min()

max_lon = df['pickup_longitude'].max()



print(min_lat, max_lat)

print(min_lon, max_lon)
west, south, east, north = -74.26, 40.50, -73.70, 40.92

df = df[(df['pickup_longitude'] > west) & (df['pickup_longitude'] < east)]

df = df[(df['pickup_latitude'] > south) & (df['pickup_latitude'] < north)]



fig = plt.figure(figsize=(14,8))



m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='c')

x, y = m(df['pickup_longitude'].values, df['pickup_latitude'].values)

m.hexbin(x, y, gridsize=300, bins='log', cmap=cm.YlOrRd_r, lw=0.4)



plt.title('Pickup Locations')

plt.show()
west, south, east, north = -74.26, 40.50, -73.70, 40.92

df = df[(df['dropoff_longitude'] > west) & (df['dropoff_longitude'] < east)]

df = df[(df['dropoff_latitude'] > south) & (df['dropoff_latitude'] < north)]



fig = plt.figure(figsize=(14,8))



m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='c')

x, y = m(df['dropoff_longitude'].values, df['dropoff_latitude'].values)

m.hexbin(x, y, gridsize=300, bins='log', cmap=cm.YlOrRd_r, lw=0.4)



plt.title('Dropoff Locations')

plt.show()
coords = np.vstack((df[['pickup_latitude',  'pickup_longitude']].values,

                    df[['dropoff_latitude', 'dropoff_longitude']].values,

                    df[['pickup_latitude',  'pickup_longitude']].values,

                    df[['dropoff_latitude', 'dropoff_longitude']].values))

kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords)
def derive_features(df):

    df = df.copy()

    # Pickup-based features

    df['pickup_date'] = df['pickup_datetime'].dt.date

    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday

    df['pickup_hour'] = df['pickup_datetime'].dt.hour

    df['pickup_month'] = df['pickup_datetime'].dt.month

    df['pickup_day'] = df['pickup_datetime'].dt.day

    # Brand new features

    df['haversine_distance'] = df.apply(lambda x: haversine((x['pickup_latitude'],  x['pickup_longitude']), 

                                                            (x['dropoff_latitude'], x['dropoff_longitude'])), axis=1)

    df['pickup_cluster']  = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])

    df['dropoff_cluster'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']])

    return df
df = derive_features(df)
df.head()
json_data = geojson.load(open('../input/pediacitiesnycneighborhoods/community-districts-polygon.geojson'))

json_data.keys()

polygons = json_data['features']

FILL = '#6699cc'

CONT = '#1a169e'
fig = plt.figure(figsize=(14,8))

ax = fig.gca() 



for i in range(len(polygons)):

    coordlist = polygons[i]['geometry']['coordinates']

    poly = {'type':'Polygon', 'coordinates':coordlist}

    ax.add_patch(PolygonPatch(poly, fc=FILL, ec=CONT, alpha=0.5, zorder=1))

    ax.axis('scaled')



ax.scatter(df['pickup_longitude'].values, df['pickup_latitude'], s=10, lw=0,

           c=df['pickup_cluster'].values, cmap='tab20', zorder=2)

plt.title('Pickup Clusters')

plt.grid(False)
fig = plt.figure(figsize=(14,8))

ax = fig.gca() 



for i in range(len(polygons)):

    coordlist = polygons[i]['geometry']['coordinates']

    poly = {'type':'Polygon', 'coordinates':coordlist}

    ax.add_patch(PolygonPatch(poly, fc=FILL, ec=CONT, alpha=0.5, zorder=1))

    ax.axis('scaled')



ax.scatter(df['dropoff_longitude'].values, df['dropoff_latitude'], s=10, lw=0,

           c=df['dropoff_cluster'].values, cmap='tab20', zorder=2)

plt.title('Dropoff Clusters')

plt.grid(False)
plt.figure(figsize=(14, 5))

plt.title('Vendor ID')

sns.countplot(x='vendor_id', data=df)
plt.figure(figsize=(14, 5))

plt.title('Trip Duration in Seconds')

sns.distplot(df['trip_duration'], hist=False)
print('Mean trip duration (min): {0:.2f}'.format(df['trip_duration'].mean()/60))

print('Max trip duration (min): {0:.2f}'.format(df['trip_duration'].max()/60))

print('Median trip duration (min): {0:.2f}'.format(df['trip_duration'].median()/60))
plt.figure(figsize=(14, 5))

plt.title('Distance Traveled in km')

sns.distplot(df['haversine_distance'], hist=False)
print('Mean trip distance (km): {0:.2f}'.format(df['haversine_distance'].mean()))

print('Max trip distance (km) {0:.2f}'.format(df['haversine_distance'].max()))

print('Median trip distance (km): {0:.2f}'.format(df['haversine_distance'].median()))
plt.figure(figsize=(14, 5))

plt.title('Pickup per Weekday')

sns.countplot(x='pickup_weekday', data=df)
plt.figure(figsize=(14, 5))

plt.title('Pickup per Hour')

sns.countplot(x='pickup_hour', data=df)
plt.figure(figsize=(14, 5))

plt.title('Passenger')

sns.countplot(x='passenger_count', data=df)
plt.figure(figsize=(14, 5))

plt.title('Pickups per Month')

sns.countplot(x='pickup_month', data=df)
plt.figure(figsize=(14, 5))

plt.title('Duration per Weekday')

sns.pointplot(x='pickup_weekday', y='trip_duration', hue='vendor_id', data=df)
plt.figure(figsize=(14, 5))

plt.title('Duration per Hour')

sns.pointplot(x='pickup_hour', y='trip_duration', hue='vendor_id', data=df)
plt.figure(figsize=(14, 5))

plt.title('Duration per Month')

sns.pointplot(x='pickup_month', y='trip_duration', hue='vendor_id', data=df)
plt.figure(figsize=(14, 5))

plt.title('Duration per Passenger Count')

sns.pointplot(x='passenger_count', y='trip_duration', hue='vendor_id', data=df)
def clean_data(df):

    df = df.copy()

    df = df[(df['trip_duration'] > 60) & (df['trip_duration'] < 3600 * 24)] # Trip duration filtering

    df = df[(df['haversine_distance'] > 0.01)] # Distance filtering

    return df
df = clean_data(df)
plt.figure(figsize=(14, 5))

plt.title('Distance Traveled in km')

sns.distplot(df['haversine_distance'], hist=False)
plt.figure(figsize=(14, 5))

plt.title('Trip Duration in Seconds')

sns.distplot(df['trip_duration'], hist=False)
print('Min trip distance (km): {0:.2f}'.format(df['haversine_distance'].min()))

print('Min trip duration (s): {0:.2f}'.format(df['trip_duration'].min()))
wf = wf.replace(to_replace='T', value=0.01)



wf['precipitation'] = pd.to_numeric(wf['precipitation'])

wf['snow fall'] = pd.to_numeric(wf['snow fall'])

wf['snow depth'] = pd.to_numeric(wf['snow depth'])



wf.tail()
df['pickup_date']   = pd.to_datetime(df['pickup_date'])

mf = df.merge(wf, left_on='pickup_date', right_on='date', how='inner')

mf.tail()
mf.dtypes
corr = mf.select_dtypes(include=[np.number]).corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True



plt.figure(figsize=(10, 10))

with sns.axes_style('white'):

    sns.heatmap(corr, mask=mask, linewidths=0.01, square=True, linecolor='white')

plt.xticks(rotation=90)

plt.title('Correlation between features')
mf = mf.merge(hf, left_on='pickup_date', right_on='Date', how='left')

mf.head()
mf.loc[~mf['Holiday'].isnull(), 'Holiday'] = 1 

mf.loc[mf['Holiday'].isnull(), 'Holiday'] = 0



mf['Holiday'] = pd.to_numeric(mf['Holiday'])



mf['Holiday'].value_counts()
mf = mf.drop(['Day', 'Date', 'date', 'pickup_date'], axis=1)
mf.dtypes
le = LabelEncoder()



for cat in ['store_and_fwd_flag', 'vendor_id']:

    mf[cat] = le.fit_transform(mf[cat])
X = mf.select_dtypes(include=[np.number])

X = X.drop(['trip_duration'], axis=1)

y = np.log1p(mf['trip_duration'])
def rmsle_eval(y, y0):

    y0 = y0.get_label()    

    assert len(y) == len(y0)

    return 'error', np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

dtrain = xgb.DMatrix(X_train, label=y_train)

dvalid = xgb.DMatrix(X_valid, label=y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



params = {}

params["objective"] = "reg:linear"

params["eta"] = 0.1

params["min_child_weight"] = 30

params["subsample"] = 0.8

params["colsample_bytree"] = 0.3

params["scale_pos_weight"] = 1.0

params["silent"] = 1

params["max_depth"] = 10

params["nthread"] = -1



plst = list(params.items())



model = xgb.train(plst, dtrain, 200, watchlist, early_stopping_rounds=50, maximize=False, 

                  verbose_eval=20, feval=rmsle_eval)
def plot_importances(clf):

    importances = clf.get_fscore()

    importances = sorted(importances.items(), key=lambda x: x[1])

    x = list(zip(*importances))[0]

    y = list(zip(*importances))[1]

    x_pos = np.arange(len(x)) 

    plt.figure(figsize=(10,5))

    plt.title('Feature importances')

    plt.barh(range(len(y)), y, align='center')

    plt.yticks(range(len(y)), x)

    plt.ylim([-1, len(importances)])

    plt.xlabel('F score')

    plt.show()
plot_importances(model)
tf = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')

ids = tf['id']

tf['pickup_datetime']  = pd.to_datetime(tf['pickup_datetime'])
tf = derive_features(tf)
tf['pickup_date']   = pd.to_datetime(tf['pickup_date'])

tf = tf.merge(wf, left_on='pickup_date', right_on='date', how='inner')
tf = tf.merge(hf, left_on='pickup_date', right_on='Date', how='left')

tf.loc[~tf['Holiday'].isnull(), 'Holiday'] = 1 

tf.loc[tf['Holiday'].isnull(), 'Holiday'] = 0



tf['Holiday'] = pd.to_numeric(tf['Holiday'])



tf['Holiday'].value_counts()
le = LabelEncoder()



for cat in ['store_and_fwd_flag', 'vendor_id']:

    tf[cat] = le.fit_transform(tf[cat])
tf = tf.drop(['Day', 'Date', 'date', 'pickup_date'], axis=1)
tf = tf.select_dtypes(include=[np.number])
dtest = xgb.DMatrix(tf)

ytest = model.predict(dtest)
submission = pd.DataFrame({'id': ids, 'trip_duration': np.expm1(ytest)})

submission.to_csv('nyc-output.csv', index=False)
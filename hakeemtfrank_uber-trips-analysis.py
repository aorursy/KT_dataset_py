import numpy as np 

import pandas as pd 





import seaborn as sns

import matplotlib.pyplot as plt



plt.style.use('seaborn')



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



plt.rcParams['figure.figsize'] = (12, 6)
taxidf = pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-janjune-15.csv")

taxilookup = pd.read_csv("/kaggle/input/rawdata/uber-tlc-foil-response-master/uber-trip-data/taxi-zone-lookup.csv")
taxidf.head()
taxidf.info()
taxidf['Pickup_date'] = pd.to_datetime(taxidf['Pickup_date'], infer_datetime_format=True)
taxilookup.head()
taxilookup.index = taxilookup['LocationID']
taxilookup.head()
taxilookup['LocationID'].nunique()
taxilookup['Borough'].nunique()
taxilookup['Zone'].nunique()
taxidf['Borough'].nunique()
taxidf['Borough'] = taxidf['locationID'].map(taxilookup['Borough'])
taxidf['Zone'] = taxidf['locationID'].map(taxilookup['Zone'])
taxidf['locationID'].value_counts().head(10)
taxidf['Borough'].value_counts().head(10)
taxidf[taxidf['Zone'].isin(taxidf['Zone'].value_counts().head(10).index.tolist())]['Borough'].value_counts()
taxidf['locationID'].value_counts().head(10).sort_values().plot(kind = 'barh')

plt.title("Top 10 Locations by Number of Trips", size = 14)

plt.show()
taxilookup[taxilookup['LocationID'] == 161]
taxilookup[taxilookup['Borough'] == 'Manhattan']
dumbo_ts = taxidf[taxidf['locationID'] == 66]
dumbo_ts.info()
dumbo_ts.head()
dumbo_ts.info()
dumnew = dumbo_ts.groupby('Pickup_date')['Pickup_date'].count()
dum_hour = dumnew.resample('6H').sum()

dum_day = dumnew.resample('D').sum()
dum_day.plot()

plt.title("Daily Taxi Pickups in Dumbo, NYC", size = 14)

plt.show()
dum_jan = dum_hour[(dum_hour.index > '2015-01-01') & (dum_hour.index < '2015-02-01')]

dum_jan.plot()

plt.title("Hourly Uber Pickups, Jan 2015 - Feb 2015", size = 14)

plt.show()
fig, axes = plt.subplots(3, 1, figsize=(14,12))

dum_hour.plot(ax = axes[0])

axes[0].set_title("Hourly Uber Pickups, Jan 2015 - July 2015", size = 14)

dum_day.plot(ax = axes[1])

axes[1].set_title("Daily Uber Pickups, Jan 2015 - July 2015", size = 14)

dum_jan.plot(ax = axes[2])

axes[2].set_title("Hourly Uber Pickups, Jan 2015 - Feb 2015", size = 14)

plt.tight_layout()

plt.show()
dum_hour.shape
top10 = taxidf['locationID'].value_counts().head(10).index.tolist()
taxidf[taxidf['locationID'].isin(top10)]
fig, axes = plt.subplots(10, 1, figsize = (15,20))

i = 0

for location in top10:

    locdf = taxidf[(taxidf['locationID'] == location)]

    locts = locdf.groupby('Pickup_date')['Pickup_date'].count()

    locts_hour = locts.resample('H').sum()

    locts_hour = locts_hour[(locts_hour.index < '2015-02-01')]

    axes[i].plot(locts_hour)

    axes[i].set_title("Location: {}".format(location))

    i += 1

plt.tight_layout()

plt.show()
bottom10 = taxidf['locationID'].value_counts().tail(10).index.tolist()
fig, axes = plt.subplots(10, 1, figsize = (15,20))

i = 0

for location in bottom10:

    locdf = taxidf[(taxidf['locationID'] == location)]

    locts = locdf.groupby('Pickup_date')['Pickup_date'].count()

    locts_hour = locts.resample('H').sum()

    locts_hour = locts_hour[(locts_hour.index < '2015-02-01')]

    axes[i].plot(locts_hour)

    axes[i].set_title("Location: {}".format(location))

    i += 1

plt.tight_layout()

plt.show()
# check random zones 

zones = [39, 91, 220, 127, 157, 72, 121, 247, 177, 71]

fig, axes = plt.subplots(10, 1, figsize = (15,20))

i = 0

for location in zones:

    locdf = taxidf[(taxidf['locationID'] == location)]

    locts = locdf.groupby('Pickup_date')['Pickup_date'].count()

    locts_hour = locts.resample('H').sum()

    locts_hour = locts_hour[(locts_hour.index < '2015-02-01')]

    axes[i].plot(locts_hour)

    axes[i].set_title("Location: {}".format(location))

    i += 1

plt.tight_layout()

plt.show()
pickup_freqs = taxidf['locationID'].value_counts().values
pickup_freqs.shape
plt.figure(figsize=(10,4))

sns.distplot(pickup_freqs)

plt.title("Number of pickups", size=14)

plt.show()
pickup_counts = taxidf['locationID'].value_counts()
pickup_counts.index
# check random zones 

zones = [113, 114, 79, 249, 107, 234, 90, 211]

fig, axes = plt.subplots(8, 1, figsize = (15,20))

i = 0

for location in zones:

    locdf = taxidf[(taxidf['locationID'] == location)]

    locts = locdf.groupby('Pickup_date')['Pickup_date'].count()

    locts_hour = locts.resample('H').sum()

    locts_hour = locts_hour[(locts_hour.index < '2015-03-01')]

    axes[i].plot(locts_hour)

    axes[i].set_title("Location: {}".format(location))

    i += 1

plt.tight_layout()

plt.show()
previewdf = taxidf[taxidf['Pickup_date'] < '2015-03-01']
previewdf.head()
previewdf = previewdf.drop(['Dispatching_base_num','Affiliated_base_num'] , axis=1)
previewdf.head()
preview_count = previewdf.groupby(['Pickup_date','locationID'])['Pickup_date'].count()


preview_h = preview_count.groupby("locationID").resample('H',level=0).sum()
preview_df = preview_h.reset_index(name='pickups')
preview_df.index = preview_df['Pickup_date']
taxidf.head()
taxidf['locationID'].value_counts().head()
# Lower Manhattan. Edit if we want to add more zones.

# East Village, Union Sq, Lil Italy, Soho, Greenich Village N/S, LES, Gramercy

#lowerman = [79, 211, 234, 144, 113, 114, 148, 107]



# 04/07/2020 we want all zones so we took this box out

lowerman = ['Manhattan']
lower_df = taxidf[taxidf['Borough'].isin(lowerman)]
lower_df.head()
lower_df = lower_df.drop(['Dispatching_base_num','Affiliated_base_num'] , axis=1)
lower_df.head()
# 1 hour intervals

lower_1count = lower_df.groupby(['Pickup_date','Zone'])['Pickup_date'].count()

lower_1h = lower_1count.groupby("Zone").resample('H',level=0).sum()

lower_1df = lower_1h.reset_index(name='pickups')

lower_1df.index = lower_1df['Pickup_date']

lower_1df.drop('Pickup_date', axis=1, inplace=True)
# 6 hour intervals

lower_6count = lower_df.groupby(['Pickup_date','Zone'])['Pickup_date'].count()

lower_6h = lower_6count.groupby("Zone").resample('6H',level=0).sum()

lower_6df = lower_6h.reset_index(name='pickups')

lower_6df.index = lower_6df['Pickup_date']

lower_6df['Zone Formatted'] = lower_6df['Zone'].str.replace(" ", "_")

lower_6df.drop('Pickup_date', axis=1, inplace=True)
lower_6df.head()
#lower_1df.to_csv("uber_pickups_lower_manhattan_long_1h.csv")
lower_6df.to_csv("uber_pickups_lower_manhattan_long_6h.csv")
# # Make a wide dataset

# wide_1df = pd.pivot_table(lower_1df, values='pickups', columns='Zone',

#               index=lower_1df.index)
# Make a wide dataset

wide_6df = pd.pivot_table(lower_6df, values='pickups', columns='Zone Formatted',

              index=lower_6df.index)
# Check for null values

wide_6df.isna().sum().sort_values(ascending=False)
wide_6df = wide_6df.fillna(0)
#wide_1df.head()
#wide_1df.to_csv("uber_pickups_lower_manhattan_wide_1h.csv")
wide_6df.to_csv("uber_pickups_lower_manhattan_wide_6h.csv")
from sklearn.manifold import Isomap

iso = Isomap(n_components=2)

ts_isomap = iso.fit_transform(wide_6df.T)

ts_df = pd.DataFrame(ts_isomap, columns=['Component 1', 'Component 2'])

ts_df.index = wide_6df.T.index
ts_df.head()
sns.scatterplot(x = 'Component 1', y='Component 2', data=ts_df)

plt.title("Pickups Embeddings: Isomap", size=18)

plt.show()
# Just in case I am going back and forth in my notebook to test clustering parameters

if ts_df.shape[1] > 2:

    ts_df = ts_df[['Component 1', 'Component 2']]
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

Z_scaled = mm.fit_transform(ts_df)

ts_df = pd.DataFrame(Z_scaled, columns=['Component 1', 'Component 2'])

ts_df.index = wide_6df.T.index
ts_df.head()
from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture
km = KMeans(n_clusters=6, random_state=12)

gmm = GaussianMixture(n_components=6, random_state=12)
km.fit(ts_df)

gmm.fit(ts_df)
cluster_label_km = km.predict(ts_df)

cluster_label_gmm = gmm.predict(ts_df)
km.cluster_centers_
gmm.means_
ts_df['KMeansCluster'] = cluster_label_km

ts_df['GMMCluster'] = cluster_label_gmm

ts_df['KMeansCluster'] = "Cluster " + ts_df['KMeansCluster'].astype('str')

ts_df['GMMCluster'] = "Cluster " + ts_df['GMMCluster'].astype('str')
ts_df.head()
fig, axes = plt.subplots(1,2, figsize=(12,5))



axes[0].set_title("K-Means Clustered Pickup Zones", size=14)

sns.scatterplot(x='Component 1',y='Component 2', hue='KMeansCluster', data=ts_df, ax=axes[0])



axes[1].set_title("GMM Clustered Pickup Zones",size=14)

sns.scatterplot(x='Component 1', y='Component 2', hue='GMMCluster', data=ts_df,ax=axes[1])

plt.tight_layout()

plt.show()
plt.figure(figsize=(6,6))

sns.scatterplot(x='Component 1',y='Component 2', hue='KMeansCluster', data=ts_df)

sns.scatterplot(x = km.cluster_centers_[:,0], y= km.cluster_centers_[:,1], color='red')

plt.show()
nearest_zones=[]

for centroid in km.cluster_centers_:

    dists = []

    for i, zone in enumerate(ts_df.index.tolist()):

        dist = np.linalg.norm(centroid - ts_df.iloc[i][['Component 1', 'Component 2']].values)

        dists.append((zone, dist))

    nearest_zone = sorted(dists, key=lambda x: x[1])[0]

    nearest_zones.append(nearest_zone)
nearest_zones
zone_list = [zone[0] for zone in nearest_zones]
centroid_zones = ts_df[ts_df.index.isin(zone_list)]

plt.figure(figsize=(6,6))

sns.scatterplot(x = 'Component 1', y='Component 2', hue = 'KMeansCluster', data=ts_df)

sns.scatterplot(x = km.cluster_centers_[:,0], y= km.cluster_centers_[:,1], color='red', label='Cluster Centroids')

sns.scatterplot(x = 'Component 1', y='Component 2', data=centroid_zones, color='orange', label='Zone Centroids')

plt.legend(frameon=True)

plt.show()
fig, axes = plt.subplots(3, 2, figsize = (12,12))

axes = axes.flatten()

date_range = (wide_6df.index >= '2015-02-01') & (wide_6df.index < '2015-03-01')



for i, zone in enumerate(zone_list):

    clust = ts_df[ts_df.index == zone]['KMeansCluster'].values[0]

    axes[i].plot(wide_6df[date_range][zone])

    axes[i].set_title("{}: {}".format(zone, clust))

    axes[i].set_xticks([""])

plt.tight_layout()
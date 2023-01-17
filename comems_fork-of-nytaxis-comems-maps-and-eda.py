import os



from geopy.distance import geodesic



import datetime as dt

import numpy as np 

import pandas as pd 

import scipy.stats as stats



import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

%matplotlib inline



import geopandas

import cartopy

import cartopy.io.img_tiles as cimgt

#Commons parameters for Matplotlib

mpl.rcParams['axes.titlesize']=22

mpl.rcParams['axes.labelsize']=18

mpl.rcParams['legend.fontsize']=14

mpl.rcParams['legend.markerscale']=3
FILEPATH = '../input/nyc-taxi-trip-duration//train.csv'

df = pd.read_csv(FILEPATH)
df.head()
xminint, xmaxint = [-74.05, -73.85]

yminint, ymaxint = [40.65, 40.9]



xmin, xmax = [-125, -60]

ymin, ymax = [31.5, 53+15]





fig, ax = plt.subplots(1, figsize=[20, 10])

ax.plot( df['pickup_longitude'], df['pickup_latitude'], linestyle='', markersize=2, marker='o', color='blue', label='pickup');

ax.plot( df['dropoff_longitude'], df['dropoff_latitude'], linestyle='', markersize=2, marker='o', color='orange', label='dropoff');

ax.plot( [xminint, xmaxint, xmaxint, xminint, xminint], [yminint, yminint, ymaxint, ymaxint, yminint], color='black', linestyle='-' );



ax.set_title('New York City Taxi Trip')

ax.legend()



ax.set_xlim([xmin, xmax]);

ax.set_ylim([ymin, ymax]);

ax.set_aspect(1)

ax.set_xticks([])

ax.set_yticks([])



ax.plot([xmaxint, -85], [yminint, 32.5], color='black', alpha=.5)

ax.plot([xmaxint, -85], [ymaxint, 66], color='black', alpha=.5)

sizeint=.7

a = plt.axes( [.1, .145, sizeint, sizeint], frameon=True)

a.plot( df['dropoff_longitude'], df['dropoff_latitude'],

       linestyle='', markersize=.05, marker='.', alpha=.5, color='orange');

a.plot( df['pickup_longitude'], df['pickup_latitude'],

       linestyle='', markersize=.02, marker='.', alpha=.5, color='blue');





a.set_xlim([xminint, xmaxint]);

a.set_ylim([yminint, ymaxint]);

a.set_aspect(1)

a.set_xticks([])

a.set_yticks([])

plt.show()
fig= plt.figure(figsize=[20, 10])

gs = mpl.gridspec.GridSpec(1, 3)



ax = fig.add_subplot(gs[0,:-1], projection=cartopy.crs.PlateCarree())



ax.add_feature(cartopy.feature.LAND, color='whitesmoke')

ax.add_feature(cartopy.feature.OCEAN, color='white')

ax.add_feature(cartopy.feature.COASTLINE)

ax.add_feature(cartopy.feature.BORDERS, linestyle=':')

ax.add_feature(cartopy.feature.LAKES, color='white')

ax.coastlines(resolution='110m')

ax.set_extent([-130, -50, 20, 60])



size = df.shape[0]

x, y = list(df['pickup_longitude'][:size]), list(df['pickup_latitude'][:size]);

ax.scatter(x, y, s=30, c=list(df['trip_duration'][:size]), norm=LogNorm(vmin=100, vmax=1000), cmap='plasma', transform=cartopy.crs.Geodetic(), zorder=10);

ax.set_aspect('equal')





ax.set_title('New York City Taxi Trips')







stamen_terrain = cimgt.Stamen('terrain-background')

ax2 = fig.add_subplot(gs[0,2], projection=cartopy.crs.PlateCarree())

ax2.add_image(stamen_terrain, 10)



center = [-73.945, 40.76]

z = 0.1

borders = [center[0]-z, center[0]+z, center[1]-1/0.7*z, center[1]+1/0.7*z]

ax2.set_extent(borders)



sc = ax2.scatter(x, y, s=.01, alpha=.5, c=list(df['trip_duration'][:size]), norm=LogNorm(vmin=100, vmax=1000), cmap='plasma', transform=cartopy.crs.Geodetic());



ax2.set_aspect('equal')



cbar = plt.colorbar(sc, fraction=.063, pad=.05)

cbar.set_alpha(1)

cbar.draw_all()

cbar.set_label('Trip duration [in s]')

plt.show()

from shapely.geometry import Point, Polygon
fig= plt.figure(figsize=[20, 10])

ax = plt.axes( projection=cartopy.crs.PlateCarree())



stamen_terrain = cimgt.Stamen('terrain-background')

ax.add_image(stamen_terrain, 10)



center = [-73.95, 40.775]

z = 0.2

borders = [center[0]-1/0.7*2*z, center[0]+1/0.7*2*z, center[1]-1/0.7*z, center[1]+1/0.7*z]

ax.set_extent(borders)



size = df.shape[0]

x, y = list(df['pickup_longitude'][:size]), list(df['pickup_latitude'][:size]);

sc = ax.scatter(x, y, s=.2, c=list(df['trip_duration'][:size]), norm=LogNorm(vmin=10, vmax=4000), cmap='brg', transform=cartopy.crs.Geodetic());



cbar = plt.colorbar(sc, fraction=.063, pad=.05)

cbar.set_alpha(1)

cbar.draw_all()

cbar.set_label('Trip duration [in s]')



gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,

                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

gl.ylabels_right = False



plt.show()
fig= plt.figure(figsize=[20, 10])

ax = plt.axes( projection=cartopy.crs.PlateCarree())



stamen_terrain = cimgt.Stamen('terrain-background')

ax.add_image(stamen_terrain, 10)



center = [-73.95, 40.775]

z = 0.2

borders = [center[0]-1/0.7*2*z, center[0]+1/0.7*2*z, center[1]-1/0.7*z, center[1]+1/0.7*z]

ax.set_extent(borders)

gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,

                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

gl.ylabels_right = False



alpha = 0.8

zorder = 3

newark = [(-74.065, 40.65),(-73.85, 41.05), (-74.2, 41.05), (-74.2, 40.64)]

poly_newark = geopandas.GeoDataFrame(index=[0], geometry=[Polygon(newark)])

poly_newark.plot(ax=ax, facecolor='green', alpha=alpha, zorder=zorder)



manhattan = [ (-74.04, 40.69), (-73.97, 40.71), (-73.965, 40.745), (-73.93, 40.78), (-73.935, 40.84), (-73.91, 40.87), (-73.94, 40.88)]

poly_manhattan = geopandas.GeoDataFrame(index=[0], geometry=[Polygon(manhattan)])

poly_manhattan.plot(ax=ax, facecolor='orange', alpha=alpha, zorder=zorder)



kennedy = [ (-73.82, 40.63), (-73.76, 40.64), (-73.808, 40.69) ]

poly_kennedy = geopandas.GeoDataFrame(index=[0], geometry=[Polygon(kennedy)])

poly_kennedy.plot(ax=ax, facecolor='yellow', alpha=alpha, zorder=zorder)



laguardia = [ (-73.885, 40.78), (-73.875, 40.768), (-73.875, 40.765), (-73.86, 40.76), (-73.855, 40.775) ]

poly_laguardia = geopandas.GeoDataFrame(index=[0], geometry=[Polygon(laguardia)])

poly_laguardia.plot(ax=ax, facecolor='blue', alpha=alpha, zorder=zorder)



ny_borders = [-74.2, -73.75, 40.5, 41.05]

nycity = [ (ny_borders[0], ny_borders[2]), (ny_borders[1], ny_borders[2]), (ny_borders[1], ny_borders[3]), (ny_borders[0], ny_borders[3]) ]

poly_nyc = geopandas.GeoDataFrame(index=[0], geometry=[Polygon(nycity)])

poly_nyc.plot(ax=ax, facecolor='black', alpha=0.2, zorder=1)



ax.text(-74.1, 40.9, 'Newark', fontsize=22)

ax.text(-74.04, 40.7, 'Manhattan', fontsize=22)

ax.text(-73.93, 40.78, 'La Guardia', fontsize=22)

ax.text(-73.83, 40.68, 'Kennedy', fontsize=22)

ax.text(-74.04, 40.53, 'New-York', fontsize=22)





plt.show()
fig= plt.figure(figsize=[20, 10])

ax = plt.axes( projection=cartopy.crs.PlateCarree())



stamen_terrain = cimgt.Stamen('terrain-background')

ax.add_image(stamen_terrain, 10)



center = [-73.95, 40.775]

z = 0.2

borders = [center[0]-1/0.7*2*z, center[0]+1/0.7*2*z, center[1]-1/0.7*z, center[1]+1/0.7*z]

ax.set_extent(borders)

gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,

                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

gl.ylabels_right = False



alpha = 0.4

zorder = 3

poly_newark.plot(ax=ax, facecolor='green', alpha=alpha, zorder=zorder)



poly_manhattan.plot(ax=ax, facecolor='orange', alpha=alpha, zorder=zorder)



poly_kennedy.plot(ax=ax, facecolor='yellow', alpha=alpha, zorder=zorder)



poly_laguardia.plot(ax=ax, facecolor='blue', alpha=alpha, zorder=zorder)



poly_nyc.plot(ax=ax, facecolor='black', alpha=0.1, zorder=1)





size = df.shape[0]

#size = 100

x, y = list(df['pickup_longitude'][:size]), list(df['pickup_latitude'][:size]);

sc = ax.scatter(x, y, s=.2, c=list(df['trip_duration'][:size]), norm=LogNorm(vmin=10, vmax=4000), cmap='brg', transform=cartopy.crs.Geodetic());



cbar = plt.colorbar(sc, fraction=.063, pad=.05)

cbar.set_alpha(1)

cbar.draw_all()

cbar.set_label('Trip duration [in s]')



plt.show()
def create_column_nyc(df_in):

    #ny_borders = [-74.2, -73.75, 40.5, 41.05]

    poly_nyc = Polygon(nycity)

    # Using lists rather than dataframes make everything faster

    lon = list(df_in['pickup_longitude'])

    lat = list(df_in['pickup_latitude'])

    nyc = [1]*df_in.shape[0]

    # Here, better check if a point is outside 

    for i in range(df_in.shape[0]):

        if (lon[i] < ny_borders[0] ) or (lon[i] > ny_borders[1] ) or (lat[i] < ny_borders[2] ) or (lat[i] > ny_borders[3] ) :

            nyc[i] = 0

    df_in['nyc'] = nyc

    return df_in

# 0.0221% of the points are outside the area `New-York City` (322 points)



def create_column_laguardia(df_in):

    #laguardia = [ (-73.885, 40.78), (-73.875, 40.768), (-73.875, 40.765), (-73.86, 40.76), (-73.855, 40.775) ]

    poly_lag = Polygon(laguardia)

    lon = list(df_in['pickup_longitude'])

    lat = list(df_in['pickup_latitude'])

    lag = [0]*df_in.shape[0]

    for i in range(df_in.shape[0]):

        #Checking for only the points on the north of the southernmost point will make it process less data (faster)

        if lat[i] >= -73.885:

            point = Point(lon[i], lat[i])

            if point.within(poly_lag):

                lag[i] = 1

    df_in['laguardia'] = lag

    return df_in

# 2.44 % of the points are inside the area `La Guardia` (35 562 points)



def create_column_manhattan(df_in):

    #manhattan = [ (-74.04, 40.69), (-73.97, 40.71), (-73.965, 40.745), (-73.93, 40.78), (-73.935, 40.84), (-73.91, 40.87), (-73.94, 40.88)]

    poly_man = Polygon(manhattan)

    lon = list(df_in['pickup_longitude'])

    lat = list(df_in['pickup_latitude'])

    man = [0]*df_in.shape[0]

    for i in range(df_in.shape[0]):

        point = Point(lon[i], lat[i])

        if point.within(poly_man):

            man[i] = 1

    df_in['manhattan'] = man

    return df_in



def create_column_kennedy(df_in):

    #kennedy = [ (-73.82, 40.63), (-73.76, 40.64), (-73.808, 40.69) ]

    poly_ken = Polygon(kennedy)

    lon = list(df_in['pickup_longitude'])

    lat = list(df_in['pickup_latitude'])

    ken = [0]*df_in.shape[0]

    for i in range(df_in.shape[0]):

        #check only the points east of the most western point of the polygon. (less data, so way faster)

        if lon[i] >= -73.82:

            point = Point(lon[i], lat[i])

            if point.within(poly_ken):

                ken[i] = 1

    df_in['kennedy'] = ken

    return df_in



def create_column_newark(df_in):

    #newark = [(-74.065, 40.65),(-73.85, 41.05), (-74.2, 41.05), (-74.2, 40.64)]

    poly_new = Polygon(newark)

    lon = list(df_in['pickup_longitude'])

    lat = list(df_in['pickup_latitude'])

    new = [0]*df_in.shape[0]

    for i in range(df_in.shape[0]):

        #check only the points west of the most eastern point of the polygon. (less data, so way faster)

        if lon[i] <= -73.85:

            point = Point(lon[i], lat[i])

            if point.within(poly_new):

                new[i] = 1

    df_in['newark'] = new

    return df_in
create_column_nyc(df)

print('{:.3}'.format(100-df['nyc'].sum()/df.shape[0]*100),'% of the points are outside the area `New-York City` ->',df.shape[0]-df['nyc'].sum(),'points')

create_column_laguardia(df)

print('{:.3}'.format(df['laguardia'].sum()/df.shape[0]*100),'% of the points are inside the area `La Guardia` ->',df['laguardia'].sum(),'points')
create_column_manhattan(df)

print('{:.3}'.format(df['manhattan'].sum()/df.shape[0]*100),'% of the points are inside the area `Manhattan` ->',df['manhattan'].sum(),'points')
create_column_kennedy(df)

print('{:.3}'.format(df['kennedy'].sum()/df.shape[0]*100),'% of the points are inside the area `John F. Kennedy International Airport` ->',df['kennedy'].sum(),'points')
create_column_newark(df)

print('{:.3}'.format(df['newark'].sum()/df.shape[0]*100),'% of the points are inside the area `Newark` ->',df['newark'].sum(),'points')
def all_zones_in_one(df_in):

    ny = list(df_in['nyc'])

    la = list(df_in['laguardia'])

    ma = list(df_in['manhattan'])

    ke = list(df_in['kennedy'])

    ne = list(df_in['newark'])

    zone = [1]*df_in.shape[0]

    for i in range(df_in.shape[0]):

        if ny[i]==0:

            zone[i]=0

        if ma[i]==1:

            zone[i]=2

        if la[i]==1:

            zone[i]=3

        if ke[i]==1:

            zone[i]=4

        if ne[i]==1:

            zone[i]=5

    # outside NYC_zone: 0

    # inside Manhattan: 2

    # inside LaGuardia Airport: 3

    # inside Kennedy Airport: 4

    # inside Newark: 5

    # elsewhere in NYC_zone: 1

    df_in['zone'] = zone

    df_in = df_in.drop(['nyc', 'laguardia', 'manhattan', 'kennedy', 'newark'], axis=1)

    return df_in
def create_zone_columns(df_in):

    create_column_nyc(df_in)

    create_column_manhattan(df_in)

    create_column_laguardia(df_in)

    create_column_kennedy(df_in)

    create_column_newark(df_in)

    

    all_zones_in_one(df_in)

    return df_in
df = all_zones_in_one(df)
fig= plt.figure(figsize=[20, 10])

ax = plt.axes( projection=cartopy.crs.PlateCarree())



stamen_terrain = cimgt.Stamen('terrain-background')

ax.add_image(stamen_terrain, 10)



center = [-73.95, 40.775]

z = 0.2

borders = [center[0]-1/0.7*2*z, center[0]+1/0.7*2*z, center[1]-1/0.7*z, center[1]+1/0.7*z]

ax.set_extent(borders)



size = df.shape[0]

#size = 100

x, y = list(df['pickup_longitude'][:size]), list(df['pickup_latitude'][:size]);

ax.scatter(x, y, s=10, c=list(df['zone'][:size]), transform=cartopy.crs.Geodetic());



gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,

                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

gl.ylabels_right = False



plt.show()
df.head()
fig, ax = plt.subplots(1, figsize=[25,5])

plt.hist(df[(df['vendor_id']==1) & (df['trip_duration']<5000)]['trip_duration'], bins=1000, alpha=.5, density=True);

plt.hist(df[(df['vendor_id']==2) & (df['trip_duration']<5000)]['trip_duration'], bins=1000, alpha=.5, density=True);

plt.title('Histogram of the trip_duration for each vendor')

plt.xlabel('trip duration [s]')

plt.ylabel('count [normalized]')

plt.show()
df['passenger_count'].value_counts()
fig, ax = plt.subplots(1, figsize=[25,5])

pass_cat = [1, 2, 5, 3, 6, 4]

for num in pass_cat:

    plt.hist(df[(df['passenger_count']==num) & (df['trip_duration']<5000)]['trip_duration'], bins=1000, alpha=.5, density=True);



plt.title('Histogram of the trip_duration for each `passengers` type')

plt.xlabel('trip duration [s]')

plt.ylabel('count [normalized]')

plt.show()
df['store_and_fwd_flag'].value_counts()
fig, ax = plt.subplots(1, figsize=[25,5])



plt.hist(df[(df['store_and_fwd_flag']=='N') & (df['trip_duration']<5000)]['trip_duration'], bins=1000, alpha=.5, density=True);

plt.hist(df[(df['store_and_fwd_flag']=='Y') & (df['trip_duration']<5000)]['trip_duration'], bins=500, alpha=.5, density=True);



plt.title('Histogram of the trip_duration for each `store_and_fwd_flag` type')

plt.xlabel('trip duration [s]')

plt.ylabel('count [normalized]')

plt.show()
def featuresEngineering(dfin):

    dfout = dfin.copy()

    

    #Distances

    dfout['distances'] = dfout.apply(lambda x: geodesic( (x['pickup_latitude'], x['pickup_longitude']),

                (x['dropoff_latitude'], x['dropoff_longitude']) ).km, axis=1)

    

    #Dates

    dfout['pickup_datetime'] = pd.to_datetime(dfout['pickup_datetime'])

    dfout['pickup_Month'] = dfout['pickup_datetime'].dt.month

    dfout['pickup_Hour'] = dfout['pickup_datetime'].dt.hour + dfout['pickup_datetime'].dt.minute/60

    dfout['pickup_Weekdays'] = dfout['pickup_datetime'].dt.weekday

    

    weekd = list(dfout['pickup_Weekdays'])

    weeke = [0]*dfout.shape[0]

    for i in range(dfout.shape[0]):

        if weekd[i]>=5:

            weeke[i] = 1

    dfout['pickup_Weekend'] = weeke

    dfout.head()

    return dfout
%%time

df_train = featuresEngineering(df)

df_train.head()
df_train_save = df_train.copy()

df_train.head()
fig= plt.figure(figsize=[20, 10])

ax = plt.axes( projection=cartopy.crs.PlateCarree())



stamen_terrain = cimgt.Stamen('terrain-background')

ax.add_image(stamen_terrain, 10)



center = [-73.95, 40.775]

z = 0.2

borders = [center[0]-1/0.7*2*z, center[0]+1/0.7*2*z, center[1]-1/0.7*z, center[1]+1/0.7*z]

ax.set_extent(borders)



size = df_train.shape[0]

#size = 100

x, y = list(df_train['pickup_longitude'][:size]), list(df_train['pickup_latitude'][:size]);

sc = ax.scatter(x, y, s=.2, c=list(df_train['distances'][:size]), norm=LogNorm(vmin=1, vmax=10), cmap='brg', transform=cartopy.crs.Geodetic());



cbar = plt.colorbar(sc, pad=.05)

cbar.set_label('Trip distances [in km]')





plt.show()
fig, ax = plt.subplots(2, figsize=[20,8])

ax[0].hist( df_train['distances'] , bins=1000 );

ax[0].set_xlabel('Distances in [km]');

ax[0].set_ylabel('Count [#]');



ax[1].hist( df_train['trip_duration'] , bins=500 );

ax[1].set_xlabel('Trip Duration  in [s]');

ax[1].set_ylabel('Count [#]');
fig, ax = plt.subplots(2, figsize=[20,8])

ax[0].hist( np.log1p(np.log1p( df_train['distances'] )) , bins=1000 );

#ax[0].set_ylim([0, 10000]);

ax[0].set_xlabel('Distances: log(1 + log (1 + D) ) ; D in [km]');

ax[0].set_ylabel('Count [#]');



ax[1].hist(  np.log(df_train['trip_duration'] ) , bins=500 );

#ax[1].set_xlim([0, 5000]);

ax[1].set_xlabel('Trip Duration: log(T) ; T in [s]');

ax[1].set_ylabel('Count [#]');
fig, ax = plt.subplots(2, figsize=[20,8])

ax[0].plot(df_train['distances'], df_train['trip_duration'], linestyle='', marker='.');

ax[0].set_title('Distances and durations');

ax[0].set_xlabel('Distances [km]');

ax[0].set_ylabel('Durations [100 s]');



ax[1].plot(np.log1p(np.log1p(df_train['distances'])), np.log(df_train['trip_duration']), linestyle='', marker='.');

ax[1].set_xlabel('log(1 + log(1 + Distances))');

ax[1].set_ylabel('log(Durations)');
FILEPATH = '../input/nyc-taxi-trip-duration/test.csv'

test = pd.read_csv(FILEPATH)

df_test = create_zone_columns(test)

df_test = featuresEngineering(test)

df_test.head()
df_train.head()
df_train.to_csv("train_w_zones2.csv", index=False)

df_test.to_csv("test_w_zones2.csv", index=False)
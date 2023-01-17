import seaborn as sns
import pandas as pd
import numpy as np

from gc import collect

from mpl_toolkits.basemap import Basemap
from matplotlib import patheffects as path_effects
import matplotlib.pyplot as plt

from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.cluster import KMeans
data = pd.read_csv('../input/Traffic_Violations.csv',
                   usecols=['Longitude',
                            'Latitude',
                            'Accident',
                            'Belts',
                            'Personal Injury',
                            'Property Damage',
                            'Fatal',
                            'Commercial License'])

data.info()
data.head()
for column in data.drop(columns=['Longitude', 'Latitude']).columns:
    data[column] = data[column].replace(['Yes', 'No'], [1, 0]).astype('uint8')
collect()
data.info()
data.describe()
data.drop(columns=['Accident'], inplace=True)
sns.heatmap(data.drop(columns=['Longitude', 'Latitude']).corr(), cmap='PuBuGn', annot=True)
data = data[(-79.4772089<=data['Longitude']) & (data['Longitude']<=-75.049228) & (37.912465<=data['Latitude']) & (data['Latitude']<=39.7210786)]
sns.distplot(data['Longitude'])
sns.distplot(data['Latitude'])
data = data[(-77.5<=data['Longitude']) & (data['Longitude']<=-76.9) & (38.93<=data['Latitude']) & (data['Latitude']<=39.35)]
lon_min, lon_max, lat_min, lat_max = (data['Longitude'].min(), data['Longitude'].max(), data['Latitude'].min(), data['Latitude'].max())
lon_center, lat_center = ((lon_min+lon_max)/2, (lat_min+lat_max)/2)

plt.subplots(figsize=(15,15))

m = Basemap(projection='merc', llcrnrlon=lon_min, urcrnrlon=lon_max, llcrnrlat=lat_min, urcrnrlat=lat_max,lon_0=lon_center, lat_0=lat_center)

# draw parallels
parallels = np.arange(lat_min,lat_max,(lat_max-lat_min)/30)
m.drawparallels(parallels,labels=[1,0,0,0])

# draw meridians
meridians = np.arange(lon_min,lon_max,(lon_max-lon_min)/30)
m.drawmeridians(meridians,labels=[0,0,0,1], rotation=90)

m.scatter(data['Longitude'].values, data['Latitude'].values, marker='o', alpha=0.005, color='red', latlon=True, s=50)
N_CLUSTERS = 500

del m
collect()

data['cluster'] = KMeans(n_clusters=N_CLUSTERS, n_init=1, max_iter=50, random_state=42).fit_predict(data[['Longitude', 'Latitude']])
data_groupby_cluster = data.groupby('cluster')
data_by_cluster = data_groupby_cluster.sum().reset_index(drop=True)
data_by_cluster['Longitude'] = data_groupby_cluster['Longitude'].mean()
data_by_cluster['Latitude'] = data_groupby_cluster['Latitude'].mean()
del data_groupby_cluster
def plot_contour_map(data, lon_col, lat_col, var_col, cmap, extent, n_bins, top):
    
    ###### PLOTTING THE MAP ITSELF ######
    
    lon_min, lon_max, lat_min, lat_max = extent
    lon_center, lat_center = ((lon_min+lon_max)/2, (lat_min+lat_max)/2)
    
    plt.subplots(figsize=(15, 15))

    m = Basemap(projection='merc',
                llcrnrlon=lon_min,
                llcrnrlat=lat_min,
                urcrnrlon=lon_max,
                urcrnrlat=lat_max,
                lon_0=lon_center,
                lat_0=lat_center
               )
    
    # draw parallels
    parallels = np.arange(lat_min,lat_max,(lat_max-lat_min)/30)
    m.drawparallels(parallels,labels=[1,0,0,0])

    # draw meridians
    meridians = np.arange(lon_min,lon_max,(lon_max-lon_min)/30)
    m.drawmeridians(meridians,labels=[0,0,0,1], rotation=90)
    
    ###### COMPUTING VALUES ALL OVER THE PLANE ######
    
    rnn = RadiusNeighborsRegressor(weights='distance')
    rnn.fit(data[[lon_col, lat_col]], data[var_col])
    
    df = pd.DataFrame(columns=['lon', 'lat', 'i', 'j'])
    
    lons_list = []
    lats_list = []
    i_list = []
    j_list = []
    
    lons, lats = m.makegrid(n_bins, n_bins)
    Z = np.empty(lons.shape)
    for i in range(lons.shape[0]):
        for j in range(lons.shape[1]):
            lons_list.append(lons[i,j])
            lats_list.append(lats[i,j])
            i_list.append(i)
            j_list.append(j)
    
    df = df.append(pd.DataFrame({ 'lon':lons_list, 'lat':lats_list, 'i':i_list, 'j':j_list }))
    df['val'] = rnn.predict(df[['lon', 'lat']])
    
    for i,j,val in zip(df['i'], df['j'], df['val']):
        Z[i,j] = val
    
    ###### PLOTTING CONTOURS ######
    
    n_levels = 2*n_bins
    
    cs1 = m.contour(lons, lats, Z, n_levels, latlon=True, linewidths=0)
    cs2 = m.contourf(lons, lats, Z, cs1.levels, extend='both', cmap=cmap, latlon=True)
    cbar = m.colorbar(cs2)
    cbar.set_label(var_col+' violations')
    
    ###### LABELLING PEAKS AND PRINTING THEIR CHARACTERISTICS ######
    
    df = df.sort_values('val', ascending=False).reset_index(drop=True).head(n_bins)
    df.reset_index(inplace=True)
    
    df['min_dist'] = np.inf
    
    for index in df.index[1:]:
        min_dist = np.inf
        for i in range(index):
            dist = ((df.at[i, 'lon']-df.at[index, 'lon'])**2+(df.at[i, 'lat']-df.at[index, 'lat'])**2)**0.5
            if dist<min_dist:
                min_dist=dist
        df.loc[df['index']==index, 'min_dist'] = min_dist
    
    df = df[df['min_dist']>0.01].reset_index(drop=True)
    
    df['x'], df['y'] = m(df['lon'].values, df['lat'].values)
    for index in df.head(top).index:
        x, y, lon, lat, val = df.at[index, 'x'], df.at[index, 'y'], df.at[index, 'lon'], df.at[index, 'lat'], df.at[index, 'val']
        text = plt.text(x, y, ' '+str(index+1), color='white', fontweight='bold')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
        print('{}: coords={}; {} violations: {}'.format(index+1, (lon,lat), var_col, val))
    
    ###### FINISHING UP AND CLEANING THE HOUSE ######
    
    plt.title(var_col, fontsize=30)
    
    del m, rnn, lons, lats, Z, cs1, cs2, df, lons_list, lats_list, i_list, j_list
    collect()
plot_contour_map(
    data=data_by_cluster,
    lon_col='Longitude',
    lat_col='Latitude',
    var_col='Belts',
    cmap=plt.cm.jet,
    extent=(lon_min, lon_max, lat_min, lat_max),
    n_bins=200,
    top=20
)
plot_contour_map(
    data=data_by_cluster,
    lon_col='Longitude',
    lat_col='Latitude',
    var_col='Personal Injury',
    cmap=plt.cm.jet,
    extent=(lon_min, lon_max, lat_min, lat_max),
    n_bins=200,
    top=10
)
plot_contour_map(
    data=data_by_cluster,
    lon_col='Longitude',
    lat_col='Latitude',
    var_col='Property Damage',
    cmap=plt.cm.jet,
    extent=(lon_min, lon_max, lat_min, lat_max),
    n_bins=200,
    top=6
)
plot_contour_map(
    data=data_by_cluster,
    lon_col='Longitude',
    lat_col='Latitude',
    var_col='Fatal',
    cmap=plt.cm.jet,
    extent=(lon_min, lon_max, lat_min, lat_max),
    n_bins=200,
    top=25
)
plot_contour_map(
    data=data_by_cluster,
    lon_col='Longitude',
    lat_col='Latitude',
    var_col='Commercial License',
    cmap=plt.cm.jet,
    extent=(lon_min, lon_max, lat_min, lat_max),
    n_bins=200,
    top=15
)
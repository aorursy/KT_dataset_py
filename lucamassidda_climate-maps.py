import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim
locator = Nominatim(user_agent="myGeocoder")
location = locator.geocode("Champ de Mars, Paris, France")
print(location.latitude, location.longitude)
location = locator.geocode("porto, Cagliari, Italy")
print(location.latitude, location.longitude)
help(Basemap)
plt.figure(figsize=(16, 8))
#m = Basemap(projection='ortho', resolution='l', lat_0=-39, lon_0=189)
m = Basemap(projection='cyl', resolution=None, lat_0=39, lon_0=9)
m.bluemarble(scale=0.5);
temperatures = pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv')
temperatures['dt'] = pd.to_datetime(temperatures['dt'])
country = 'France'
print(temperatures.loc[temperatures.Country == country].City.unique())
city = 'Marseille'
temperatures.loc[temperatures.City == city].head()
print(city + ', ' + country)
location = locator.geocode(city + ', ' + country)
print(location.latitude, location.longitude)
lat = location.latitude
lon = location.longitude
print('Latitudine di ' + city + ': ' + str(lat))
print('Longitudine di ' + city + ': ' + str(lon))
plt.figure(figsize=(16, 8));
m = Basemap(projection='lcc', resolution='l',
            width=4E6, height=2E6, 
            lat_0=lat, lon_0=lon,)
#m.etopo(scale=1, alpha=1)
#m.bluemarble(scale=1, alpha=1)
m.drawcoastlines()
m.drawcountries()

# Map (long, lat) to (x, y) for plotting
x, y = m(lon, lat)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, city, fontsize=12);
print(lat, lon)
plt.figure(figsize=(16, 8))
m = Basemap(projection='lcc', resolution=None,
            width=4E6, height=2E6, 
            lat_0=lat, lon_0=lon,)

# draw a shaded-relief image
m.shadedrelief(scale=1)

# lats and longs are returned as a dictionary
lats = m.drawparallels(np.linspace(-90, 90, 19))
lons = m.drawmeridians(np.linspace(-180, 180, 37))

x, y = m(lon, lat)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, city, fontsize=12);
dt = pd.to_datetime('2002-07-01')
temperatures.head()
print(country)
print(dt)
temperatures.loc[(temperatures.Country == country) & (temperatures.dt == dt)]
city_name = temperatures.loc[(temperatures.Country == country) & (temperatures.dt == dt)].City.values
city_temp = temperatures.loc[(temperatures.Country == country) & (temperatures.dt == dt)].AverageTemperature.values
print(city_name)
print(city_temp)
city_lat = np.zeros(len(city_name))
city_lon = np.zeros(len(city_name))
city_lat
for i in range(len(city_name)):
    location = locator.geocode(city_name[i] + ', ' + country)
    city_lat[i] = location.latitude
    city_lon[i] = location.longitude
    print(i, city_name[i], city_temp[i], city_lat[i], city_lon[i])
city_lat
city_temp
plt.figure(figsize=(16, 8))
m = Basemap(projection='lcc', resolution='l',
            width=4E6, height=2E6, 
            lat_0=city_lat.mean(), lon_0=city_lon.mean(),)

# draw a shaded-relief image
m.shadedrelief(scale=1)
m.drawcoastlines()
m.drawcountries()
#m.bluemarble(scale=1)

# lats and longs are returned as a dictionary
lats = m.drawparallels(np.linspace(-90, 90, 19))
lons = m.drawmeridians(np.linspace(-180, 180, 37))

#for i in range(len(city_name)):
#    x, y = m(city_lon[i], city_lat[i])
#    plt.plot(x, y, 'ok', markersize=5)
#    plt.text(x, y, city_name[i], fontsize=12);

m.scatter(city_lon, city_lat, latlon=True,
          c=city_temp, s=100., cmap='Reds');
#          cmap='Reds', alpha=0.5)

# 3. create colorbar and legend
plt.colorbar();

temperatures.loc[(temperatures.City == city_name[0]) & 
                 (temperatures.dt < pd.to_datetime('2010-01-01')) &
                 (temperatures.dt >= pd.to_datetime('1980-01-01'))].AverageTemperature.mean()
print(city_name[0])
city_temp_1980_2010 = np.zeros(len(city_name))
city_temp_1950_1980 = np.zeros(len(city_name))
for i in range(len(city_name)):
    city_temp_1980_2010[i] = temperatures.loc[(temperatures.City == city_name[i]) & 
                                              (temperatures.dt < pd.to_datetime('2010-01-01')) &
                                              (temperatures.dt >= pd.to_datetime('1980-01-01'))].AverageTemperature.mean()
    city_temp_1950_1980[i] = temperatures.loc[(temperatures.City == city_name[i]) & 
                                              (temperatures.dt < pd.to_datetime('1980-01-01')) &
                                              (temperatures.dt >= pd.to_datetime('1950-01-01'))].AverageTemperature.mean()
    print(city_name[i], city_temp_1950_1980[i], city_temp_1980_2010[i])
city_delta = city_temp_1980_2010 - city_temp_1950_1980
print(city_delta)
for i in range(len(city_name)):
    print(city_name[i], city_delta[i])
plt.figure(figsize=(16, 8))
m = Basemap(projection='lcc', resolution='l',
            width=4E6, height=2E6, 
            lat_0=city_lat.mean(), lon_0=city_lon.mean(),)

# draw a shaded-relief image
m.shadedrelief(scale=1)
m.drawcoastlines()
m.drawcountries()

# lats and longs are returned as a dictionary
lats = m.drawparallels(np.linspace(-90, 90, 19))
lons = m.drawmeridians(np.linspace(-180, 180, 37))

m.scatter(city_lon, city_lat, latlon=True,
          c=city_delta, s=100.,);
#          cmap='Reds', alpha=0.5)

# 3. create colorbar and legend
plt.colorbar();


PrimaryBiomes_names=['Humid',
                     'Intermediate',
                     'Semi-arid',
                     'Desert']

Biomes_names=['Tropical Rainforest',
              'Temperate Rainforest', 
              'Broadleaf Forest',
              'Mixed Boreal-Broadleaf Forest',
              'Boreal Forest',
              'Moist Tundra or Alpine', 
              'Tall Grass Prairie',
              'Short Grass Prairie',
              'Steppe',
              'Cool Steppe',
              'Boreal Forest',
              'Dry Tundra or Alpine',
              'Forest-Tundra Transition',
              'Polar Desert',
              'Low Latitude Desert']
len(Biomes_names)
len(PrimaryBiomes_names)
lat = np.load('../input/clima-globale/lat.npy')
lon = np.load('../input/clima-globale/lon.npy')
lsmask = np.load('../input/clima-globale/lsmask.npy')
Tclima = np.load('../input/clima-globale/Tclima2020.npy')
Pclima = np.load('../input/clima-globale/Pclima2020.npy')
len(lat)
Pclima.shape
lsmask.shape
plt.figure(figsize=(16,6))
for m in range(12):
    plt.subplot(3,4,m+1)
    plt.imshow(Tclima[m],extent=[lon[0], lon[-1],lat[-1],lat[0]], cmap='rainbow')
    if(m<4) : plt.title('Temp. monthly mean [C]');
plt.figure(figsize=(16,6))
for m in range(12):
    plt.subplot(3,4,m+1)
    plt.imshow(Pclima[m],extent=[lon[0], lon[-1],lat[-1],lat[0]], cmap='rainbow')
    if(m<4) : plt.title('Prec. montlhy mean [m/month]');
month = 0 # indici da zero quindi luglio

plt.figure(figsize=(16, 8))
m = Basemap(projection='cyl', resolution='c')
#m = Basemap(projection='ortho', resolution='l', lat_0=39, lon_0=9)


xx, yy = np.meshgrid(lon, lat);

m.pcolormesh(xx, yy, Tclima[month], cmap='rainbow', latlon=True);
#m.contourf(xx, yy, Pclima[month], cmap='Blues', latlon=True, alpha=1);
plt.colorbar();
m.drawcoastlines();
month = 6 # indici da zero quindi luglio

plt.figure(figsize=(16, 8))
m = Basemap(projection='cyl', resolution='c')
#m = Basemap(projection='ortho', resolution='l', lat_0=39, lon_0=9)


xx, yy = np.meshgrid(lon, lat);

m.contourf(xx, yy, Tclima[month], cmap='rainbow', latlon=True, alpha=1);
plt.colorbar();
m.drawcoastlines();
month = 10 # indici da zero quindi luglio

plt.figure(figsize=(16, 8))
m = Basemap(projection='cyl', resolution='c')
#m = Basemap(projection='ortho', resolution='l', lat_0=39, lon_0=9)


xx, yy = np.meshgrid(lon, lat);

m.contourf(xx, yy, Pclima[month], cmap='Blues', latlon=True, alpha=1);
plt.colorbar();
m.drawcoastlines();
Tclima.shape
144*288
nlat = len(lat)
nlon = len(lon)
nlat_2 = int(nlat / 2)

# calcola dal clima i campi che servono per individure i biomi
P_annual = np.zeros([nlat,nlon])
T_annual = np.zeros([nlat,nlon])
T_summer = np.zeros([nlat,nlon])

P_annual = np.sum(Pclima, axis=0)
T_annual = np.mean(Tclima, axis=0)
print(P_annual.shape, T_annual.shape)
T_summer[:nlat_2:,:]=(Tclima[5,:nlat_2,:]+Tclima[6,:nlat_2,:]+Tclima[7,:nlat_2,:])/3
T_summer[nlat_2:,:]=(Tclima[0,nlat_2:,:]+Tclima[1,nlat_2:,:]+Tclima[11,nlat_2:,:])/3
print(T_summer.shape)
plt.imshow(T_summer, cmap='rainbow');
plt.colorbar();
plt.figure(figsize=(16, 8))
m = Basemap(projection='cyl', resolution='c')
#m = Basemap(projection='ortho', resolution='l', lat_0=39, lon_0=9)


xx, yy = np.meshgrid(lon, lat);

m.pcolormesh(xx, yy, T_annual, cmap='rainbow', latlon=True, alpha=1);
plt.colorbar();
m.drawcoastlines();
plt.figure(figsize=(16, 8))
m = Basemap(projection='cyl', resolution='c')
#m = Basemap(projection='ortho', resolution='l', lat_0=39, lon_0=9)


xx, yy = np.meshgrid(lon, lat);

m.pcolormesh(xx, yy, T_summer, cmap='rainbow', latlon=True, alpha=1);
plt.colorbar();
m.drawcoastlines();
plt.figure(figsize=(16, 8))
m = Basemap(projection='cyl', resolution='c')
#m = Basemap(projection='ortho', resolution='l', lat_0=39, lon_0=9)


xx, yy = np.meshgrid(lon, lat);

#m.pcolormesh(xx, yy, P_annual, cmap='Blues', latlon=True, alpha=1);
m.contourf(xx, yy, P_annual, cmap='Blues', latlon=True, alpha=1);
plt.colorbar();
m.drawcoastlines();
# calcola i biomi primari
PrimaryBiomes = np.zeros([nlat,nlon])
print(PrimaryBiomes.shape)
P_annual.shape
PrimaryBiomes[P_annual>=1.5] = 1
PrimaryBiomes[(P_annual>=0.62)&(P_annual<1.5)] = 2
PrimaryBiomes[(P_annual>=0.25)&(P_annual<0.62)] = 3
PrimaryBiomes[P_annual<0.25] = 4
PrimaryBiomes.shape
for i in range(4):
    print('primary biome',i+1,(PrimaryBiomes==(i+1)).sum())
plt.imshow(PrimaryBiomes, cmap='rainbow');
plt.colorbar();
plt.figure(figsize=(16, 8))
m = Basemap(projection='cyl', resolution='c')
#m = Basemap(projection='ortho', resolution='l', lat_0=39, lon_0=9)


xx, yy = np.meshgrid(lon, lat);

m.contourf(xx, yy, PrimaryBiomes, cmap=plt.cm.get_cmap('rainbow', 4), latlon=True);
#plt.colorbar(np.arange(1,5), PrimaryBiomes_names);
cbar = plt.colorbar();
cbar.set_ticks(list())

for index, label in enumerate(PrimaryBiomes_names):
    x = 1.5
    y = (2 * index + 1) / 8
    cbar.ax.text(x, y, label)

m.drawlsmask(resolution='c', land_color='none', ocean_color='w', zorder=25);
m.drawcoastlines(zorder=26);
Biomes = np.zeros([nlat,nlon])
#Humid Biomes
Biomes[(PrimaryBiomes==1)&(T_summer>=21)]=1
Biomes[(PrimaryBiomes==1)&(T_summer>=10)&(T_summer<21)]=2
#Intermediate Biomes
Biomes[(PrimaryBiomes==2)&(T_summer>=23)]=3
Biomes[(PrimaryBiomes==2)&(T_summer>=18)&(T_summer<23)]=4
Biomes[(PrimaryBiomes==2)&(T_summer>=11)&(T_summer<18)]=5
Biomes[(PrimaryBiomes==2)&(T_summer>=0)&(T_summer<11)]=6
#Semi-ard Biomes
Biomes[(PrimaryBiomes==3)&(T_summer>18.5)&(P_annual>0.45)]=7
Biomes[(PrimaryBiomes==3)&(T_summer>18.5)&(P_annual>0.35)&(P_annual<0.45)]=8
Biomes[(PrimaryBiomes==3)&(T_summer>18.5)&(P_annual<0.35)]=9
Biomes[(PrimaryBiomes==3)&(T_summer>12)&(T_summer<18.5)&(T_annual>1)]=10
Biomes[(PrimaryBiomes==3)&(T_summer>13)&(T_summer<18.5)&(T_annual<1)]=11
Biomes[(PrimaryBiomes==3)&(T_summer<13)]=12
Biomes[(PrimaryBiomes==3)&(T_summer<13)&(T_annual<-1)]=13
#Desertic Biomes
Biomes[(PrimaryBiomes==4)&(T_annual<=0)]=14
Biomes[(PrimaryBiomes==4)&(T_annual>0)]=15
for i in range(15):
    print('biome',i+1,(Biomes==(i+1)).sum())
plt.figure(figsize=(16, 8))
m = Basemap(projection='cyl', resolution='c')
#m = Basemap(projection='ortho', resolution='l', lat_0=39, lon_0=9)


xx, yy = np.meshgrid(lon, lat);

m.contourf(xx, yy, Biomes, cmap=plt.cm.get_cmap('rainbow', 15), latlon=True, 
           vmin=1, vmax=15, levels=16);
#plt.colorbar(np.arange(1,5), PrimaryBiomes_names);
cbar = plt.colorbar();
cbar.set_ticks(list())

for index, label in enumerate(Biomes_names):
    x = 1.5
    y = (2 * index + 1) / (2*15)
    cbar.ax.text(x, y, label)

m.drawlsmask(resolution='c', land_color='none', ocean_color='w', zorder=25);
m.drawcoastlines(zorder=26);
lsmask
# checking data for cagliari
dx=360./len(lon)
lat_ca=39.2
lon_ca=9.1
i_ca=np.int((90-lat_ca)/dx)
j_ca=np.int((lon_ca)/dx)
print(i_ca,j_ca,lat[i_ca],lon[j_ca])
month=np.linspace(0.5,11.5,12)
plt.figure(figsize=(16,4));
plt.subplot(121);
plt.plot(month,Tclima[:,i_ca,j_ca]);
plt.grid();
plt.subplot(122);
plt.plot(month,Pclima[:,i_ca,j_ca]);
plt.grid();
Area=np.zeros([nlat,nlon])
dx=360./nlon*60*1.852
for i in range(nlat):
    a=np.cos(lat[i]*np.pi/180.)*dx**2
    Area[i,:]=a
plt.figure(figsize=(10,4))
plt.imshow(Area,extent=[lon[0],lon[-1],lat[-1],lat[0]])
plt.title('Area [km^2]')
plt.colorbar();
PrimaryBiomes.shape
PrimaryBiomes_names[int(PrimaryBiomes[40,7]) -1]
Biomes_names[int(Biomes[30,7]) -1]
print('Latitudine: ', lat[30])
print('Longitudine: ', lon[7])
i = 3
print(PrimaryBiomes_names[i])
mask = PrimaryBiomes==(i+1)
Area_land = (Area*lsmask)
Area_land.shape
np.sum(Area_land[mask])
for i in range(4):
    mask = (PrimaryBiomes==(i+1))
    a = np.sum(Area_land[mask])
    print(PrimaryBiomes_names[i]+' area km^2:',np.int(a))
superfici_2020 = np.zeros(15)
superfici_2100 = np.zeros(15)
for i in range(15):
    mask = (Biomes==(i+1))
    a = np.sum(Area_land[mask])
    superfici_2020[i] = a
    #superfici_2100[i] = a
    print(Biomes_names[i]+' area km^2:',np.int(a))
superfici_2020
